import os
import csv
import random
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from models  import *

caler = StandardScaler()
##########################
# 데이터 전처리
##########################
def preprocess_trajectory_data(data_list):
    # DataFrame 구성
    df_t = pd.DataFrame(data_list, columns=[
        'r', 'sequence', 'timestamp', 'deg', 'deg/sec', 'mA',
        'endpoint', 'grip/rotation', 'torque', 'force', 'ori', '#'
    ])
    
    # 필터링: r != 's'
    df_t = df_t[df_t['r'] != 's']
    data_v = df_t.drop(['r', 'grip/rotation', '#'], axis=1)
    
    # 각 컬럼 split 처리
    splits = {
        'endpoint': ['x_end', 'y_end', 'z_end'],
        'deg': ['deg1', 'deg2', 'deg3', 'deg4'],
        'deg/sec': ['degsec1', 'degsec2', 'degsec3', 'degsec4'],
        'torque': ['torque1', 'torque2', 'torque3'],
        'force': ['force1', 'force2', 'force3'],
        'ori': ['yaw', 'pitch', 'roll']
    }
    
    for col, new_cols in splits.items():
        v_split = data_v[col].astype(str).str.split('/')
        for idx, new_col in enumerate(new_cols):
            data_v[new_col] = v_split.str.get(idx)
    
    # 원본 컬럼 제거
    data_v = data_v.drop(['deg', 'deg/sec', 'mA', 'endpoint', 'torque', 'force', 'ori'], axis=1)
    
    # 숫자 변환
    data_v = data_v.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # time 열 생성
    data_v['time'] = data_v['timestamp'] - data_v['sequence'] - 1
    data_v = data_v.drop(['sequence', 'timestamp'], axis=1)
    
    # 시간 기준 정렬
    data_v.sort_values(by=["time"], ascending=True, inplace=True)
    data_v.reset_index(drop=True, inplace=True)
    
    return data_v

##########################
# 분류
##########################
def load_and_preprocess_trajectory(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)
    df_t = pd.DataFrame(data_list, columns=[
        'r', 'sequence', 'timestamp', 'deg', 'deg/sec', 'mA',
        'endpoint', 'grip/rotation', 'torque', 'force', 'ori', '#'
    ])

    # 필터링
    df_t = df_t[df_t['r'] != 's']
    data_v = df_t.drop(['r', 'grip/rotation', '#'], axis=1)

    # endpoint
    v_split = data_v['endpoint'].astype(str).str.split('/')
    data_v['x_end'] = v_split.str.get(0)
    data_v['y_end'] = v_split.str.get(1)
    data_v['z_end'] = v_split.str.get(2)

    # deg
    v_split = data_v['deg'].astype(str).str.split('/')
    data_v['deg1'] = v_split.str.get(0)
    data_v['deg2'] = v_split.str.get(1)
    data_v['deg3'] = v_split.str.get(2)
    data_v['deg4'] = v_split.str.get(3)

    # deg/sec
    v_split = data_v['deg/sec'].astype(str).str.split('/')
    data_v['degsec1'] = v_split.str.get(0)
    data_v['degsec2'] = v_split.str.get(1)
    data_v['degsec3'] = v_split.str.get(2)
    data_v['degsec4'] = v_split.str.get(3)

    # torque
    v_split = data_v['torque'].astype(str).str.split('/')
    data_v['torque1'] = v_split.str.get(0)
    data_v['torque2'] = v_split.str.get(1)
    data_v['torque3'] = v_split.str.get(2)

    # force
    v_split = data_v['force'].astype(str).str.split('/')
    data_v['force1'] = v_split.str.get(0)
    data_v['force2'] = v_split.str.get(1)
    data_v['force3'] = v_split.str.get(2)

    # ori
    v_split = data_v['ori'].astype(str).str.split('/')
    data_v['yaw'] = v_split.str.get(0)
    data_v['pitch'] = v_split.str.get(1)
    data_v['roll'] = v_split.str.get(2)

    # 제거
    data_v = data_v.drop(['deg', 'deg/sec', 'mA', 'endpoint', 'torque', 'force', 'ori'], axis=1)

    # 숫자 변환
    data_v = data_v.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 'time' 열
    data_v['time'] = data_v['timestamp'] - data_v['sequence'] - 1
    data_v = data_v.drop(['sequence', 'timestamp'], axis=1)

    # 시간 정렬
    data_v.sort_values(by=['time'], ascending=True, inplace=True)
    data_v.reset_index(drop=True, inplace=True)

    # 스케일러 적용
    arr_scaled = scaler.transform(data_v.values)
    return torch.tensor(arr_scaled, dtype=torch.float32)

##########################
# 생성
##########################
class TrajectoryAnalyzer:
    """궤적 데이터 로드 및 전처리를 담당하는 클래스"""
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.golden_dir = os.path.join(base_dir, "golden_sample")
        self.non_golden_dir = os.path.join(base_dir, "non_golden_sample")

    def process_data(self, df, file_name=None):
        """DataFrame을 전처리하고 반환"""
        try:
            # 데이터프레임이 아닌 경우 (리스트 등) DataFrame으로 변환
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df, columns=[
                    'r', 'sequence', 'timestamp', 'deg', 'deg/sec', 'mA',
                    'endpoint', 'grip/rotation', 'torque', 'force', 'ori', '#'
                ])

            # 전처리 함수 호출
            processed_df = preprocess_trajectory_data(df)
            
            return processed_df

        except Exception as e:
            print(f"Error processing data from {file_name if file_name else 'unknown file'}: {str(e)}")
            return None
    
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
        
        # CSV 파일 읽기
        try:
            with open(file_path, 'r') as f:
                data_list = list(csv.reader(f))
            processed_df = self.process_data(data_list, selected_file)
            return processed_df, selected_file
        
        except Exception as e:
            print(f"Error loading {selected_file}: {str(e)}")
            return None, None
    
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
    
# 궤적 생성 및 시각화
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