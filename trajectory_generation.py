import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import math
import random

class TrajectoryAnalyzer:
    """궤적 데이터 로드 및 전처리를 담당하는 클래스"""
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.golden_dir = os.path.join(base_dir, "golden_sample")
        self.non_golden_dir = os.path.join(base_dir, "non_golden_sample")
    
    def process_data(self, df, label_name):
        """데이터 전처리"""
        # 칼럼 이름 변경
        df.columns = ['r', 'sequence', 'timestamp', 'deg', 'deg/sec', 'mA', 
                     'endpoint', 'grip/rotation', 'torque', 'force', 'ori', '#']

        df = df[df['r'] != 's']

        # 필요하지 않은 칼럼 삭제
        df = df.drop(['r', 'grip/rotation', '#'], axis=1)

        v_split = df['endpoint'].astype(str).str.split('/')
        df['x_end'] = v_split.str.get(0)
        df['y_end'] = v_split.str.get(1)
        df['z_end'] = v_split.str.get(2)

        v_split = df['ori'].astype(str).str.split('/')
        df['yaw'] = v_split.str.get(0)
        df['pitch'] = v_split.str.get(1)
        df['roll'] = v_split.str.get(2)

        for col in ['deg', 'deg/sec', 'torque', 'force']:
            v_split = df[col].astype(str).str.split('/')
            parts = v_split.apply(lambda x: pd.Series(x))
            parts.columns = [f'{col}{i+1}' for i in range(parts.shape[1])]
            df = pd.concat([df, parts], axis=1)

        # 원본 칼럼 삭제
        df = df.drop(['deg', 'deg/sec', 'mA', 'endpoint', 'torque', 'force', 'ori'], axis=1)

        # 모든 칼럼을 숫자형으로 변환
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        df['deg2'] = df['deg2'] - 90
        df['deg4'] = df['deg4'] - 90

        # 'time' 칼럼 생성 및 데이터 정렬
        df['time'] = df['timestamp'] - df['sequence'] - 1
        df = df.drop(['sequence', 'timestamp'], axis=1)
        df = df.sort_values(by=["time"], ascending=True).reset_index(drop=True)

        return df
    
    def get_random_trajectory_by_type(self, folder_path, movement_type):
        """특정 동작 타입의 궤적을 랜덤하게 선택"""
        if not os.path.exists(folder_path):
            raise ValueError(f"경로가 존재하지 않습니다: {folder_path}")
            
        # 해당 동작 타입으로 시작하는 파일들만 필터링
        trajectory_files = [f for f in os.listdir(folder_path) 
                          if f.endswith('.txt') and f.startswith(movement_type)]
        
        if not trajectory_files:
            raise ValueError(f"폴더에 {movement_type} 타입의 궤적 파일이 없습니다: {folder_path}")
            
        selected_file = random.choice(trajectory_files)
        file_path = os.path.join(folder_path, selected_file)
        
        df = pd.read_csv(file_path, delimiter=',')
        processed_df = self.process_data(df, os.path.basename(file_path))
        
        return processed_df, selected_file
    
    def get_available_movement_types(self):
        """golden_sample 디렉토리에서 사용 가능한 동작 타입 목록 반환"""
        golden_files = os.listdir(self.golden_dir)
        movement_types = set()
        
        for file in golden_files:
            if file.endswith('.txt'):
                movement_type = '_'.join(file.split('_')[:2])
                movement_types.add(movement_type)
        
        return sorted(list(movement_types))

    def load_random_trajectories(self, movement_type):
        """golden과 non-golden 샘플에서 같은 movement_type의 랜덤 궤적 선택"""
        # 각각의 폴더에서 랜덤 선택
        target_trajectory, target_file = self.get_random_trajectory_by_type(
            self.golden_dir, movement_type)
        user_trajectory, user_file = self.get_random_trajectory_by_type(
            self.non_golden_dir, movement_type)
        
        print(f"Target_Trajectory: {target_file}")
        print(f"User_Trajectory: {user_file}")
        
        return target_trajectory, user_trajectory
        

class TrajectoryGenerator:
    """궤적 생성, 변환, 시각화를 담당하는 클래스"""
    def __init__(self):
        pass

    @staticmethod
    def smooth_data(data, sigma=10):
        """가우시안 필터를 사용한 라벨 스무딩"""
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(data, sigma=sigma)

    @staticmethod
    def normalize_time(trajectory, num_points=100):
        """시간에 대해 궤적을 정규화"""
        current_length = len(trajectory)
        old_time = np.linspace(0, 1, current_length)
        new_time = np.linspace(0, 1, num_points)

        interpolator_x = interp1d(old_time, trajectory[:, 0], kind='cubic')
        interpolator_y = interp1d(old_time, trajectory[:, 1], kind='cubic')
        interpolator_z = interp1d(old_time, trajectory[:, 2], kind='cubic')

        return np.column_stack((
            interpolator_x(new_time),
            interpolator_y(new_time),
            interpolator_z(new_time)
        ))

    def apply_dtw(self, target, subject, interpolation_weight=0.5):
        """
        DTW를 적용하여 궤적을 정렬하고 보간
        
        Parameters:
            target: 타겟 궤적
            subject: 사용자 궤적
            interpolation_weight: 보간 가중치 (0: 사용자 궤적에 가깝게, 1: 타겟 궤적에 가깝게)
        """
        # 시간 정규화
        target_norm = self.normalize_time(target)
        subject_norm = self.normalize_time(subject)

        # 스무딩 적용
        target_smoothed = np.zeros_like(target_norm)
        subject_smoothed = np.zeros_like(subject_norm)
        for i in range(3):
            target_smoothed[:, i] = self.smooth_data(target_norm[:, i])
            subject_smoothed[:, i] = self.smooth_data(subject_norm[:, i])

        # DTW 거리 및 경로 계산
        distance, path = fastdtw(target_smoothed, subject_smoothed, dist=euclidean)
        path = np.array(path)

        # 매칭된 포인트들 추출
        target_matched = target_smoothed[path[:, 0]]
        subject_matched = subject_smoothed[path[:, 1]]

        # 보간된 궤적 생성
        interpolated = (target_matched * interpolation_weight + 
                       subject_matched * (1 - interpolation_weight))

        # 결과 궤적을 원본 길이로 리샘플링
        return self.normalize_time(interpolated, num_points=len(target))

    def compare_trajectories(self, target_df, user_df, save_animation=False):
        """
        타겟과 사용자 궤적을 비교하고 정렬된 궤적을 생성하여 시각화
        
        Parameters:
            target_df: 타겟 궤적 데이터프레임
            user_df: 사용자 궤적 데이터프레임
            save_animation: 애니메이션 저장 여부
        Returns:
            aligned_trajectory: DTW로 정렬된 궤적
        """
        # 원본 궤적 포인트 추출
        target_points = target_df[['x_end', 'y_end', 'z_end']].values
        user_points = user_df[['x_end', 'y_end', 'z_end']].values
        
        # DTW를 사용하여 정렬된 궤적 생성
        aligned_trajectory = self.apply_dtw(target_points, user_points)
        
        # 세 궤적 모두 시각화
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 타겟 궤적
        ax.plot(target_points[:, 0], target_points[:, 1], target_points[:, 2],
                'b--', label='Target Trajectory')
        
        # 원본 사용자 궤적
        ax.plot(user_points[:, 0], user_points[:, 1], user_points[:, 2],
                'r--', label='Original Subject Trajectory')
        
        # 정렬된 궤적
        ax.plot(aligned_trajectory[:, 0], aligned_trajectory[:, 1], aligned_trajectory[:, 2],
                'g-', label='Aligned Subject Trajectory', linewidth=2)

        # 그래프 설정
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title('Trajectory Alignment using DTW')
        ax.view_init(10, 90)
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper right')
        ax.grid(True)

        # 모든 궤적을 포함하도록 축 범위 설정
        all_points = np.vstack([target_points, user_points, aligned_trajectory])
        margin = 10  # 여백 추가
        ax.set_xlim([min(all_points[:, 0]) - margin, max(all_points[:, 0]) + margin])
        ax.set_ylim([min(all_points[:, 1]) - margin, max(all_points[:, 1]) + margin])
        ax.set_zlim([min(all_points[:, 2]) - margin, max(all_points[:, 2]) + margin])

        plt.show()
        
        return aligned_trajectory
    
# def main():
#     base_dir = os.path.join(os.getcwd(), "data")
#     # 분석기와 생성기 초기화
#     analyzer = TrajectoryAnalyzer()
#     generator = TrajectoryGenerator()
    
#     try:
#         # 사용 가능한 동작 타입 출력
#         movement_types = analyzer.get_available_movement_types()
#         print("사용 가능한 동작 타입:", movement_types)
        
#         if movement_types:
#             movement_type = random.choice(movement_types)
#             print(f"\n선택된 동작 타입: {movement_type}")
            
#             # 선택된 동작 타입의 궤적 로드
#             target_traj, user_traj = analyzer.load_random_trajectories(movement_type)
            
#             # 궤적 비교 및 새로운 궤적 생성
#             aligned_trajectory = generator.compare_trajectories(target_traj, user_traj)
#         else:
#             print("사용 가능한 동작 타입이 없습니다.")
        
#     except Exception as e:
#         print(f"오류 발생: {str(e)}")

# if __name__ == "__main__":
#     main()   
