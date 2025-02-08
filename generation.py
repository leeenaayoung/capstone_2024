import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from fastdtw import fastdtw
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.distance import euclidean
import os
import random
from analyzer import TrajectoryAnalyzer
from utils import calculate_end_effector_position

# 궤적 생성, 변환, 시각화
class TrajectoryGenerator:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.joint_limits = {
            0: (-10, 110),   # degree 제한 범위 적용
            1: (0, 150),     
            2: (0, 150),     
            3: (-90, 90)     
        }

    def smooth_data(self, data, R=0.02, Q=0.1):
        """칼만 필터를 사용한 데이터 스무딩"""
        joint_angles = data[['deg1', 'deg2', 'deg3', 'deg4']].values
        n_samples, n_joints = joint_angles.shape
        smoothed_angles = np.zeros_like(joint_angles)
        
        # 각 관절에 대해 칼만 필터 적용
        for joint in range(n_joints):
            x_hat = joint_angles[0, joint]  # 위치
            v_hat = 0  # 초기 속도
            x_hat_full = np.array([x_hat, v_hat])
            P_full = np.eye(2)  # 초기 공분산 행렬
            
            # 시스템 행렬 설정
            dt = 1.0  # 시간 간격
            A = np.array([[1, dt],
                        [0, 1]])  # 상태 변환 행렬
            H = np.array([1, 0])   # 측정 행렬
            
            # 프로세스 노이즈 공분산 행렬
            Q_matrix = Q * np.array([[dt**4/4, dt**3/2],
                                    [dt**3/2, dt**2]])
            
            # 칼만 필터 적용 (순방향)
            smoothed_angles[0, joint] = x_hat
            for k in range(1, n_samples):
                # 예측 단계
                x_hat_full = A @ x_hat_full
                P_full = A @ P_full @ A.T + Q_matrix
                
                # 업데이트 단계
                z = joint_angles[k, joint]  # 현재 측정값
                y = z - H @ x_hat_full     # 측정 잔차
                S = H @ P_full @ H.T + R   # 잔차 공분산
                K = P_full @ H.T / S       # 칼만 이득
                
                # 상태 업데이트
                x_hat_full = x_hat_full + K * y
                P_full = (np.eye(2) - np.outer(K, H)) @ P_full
                
                # 스무딩된 각도 저장
                smoothed_angles[k, joint] = x_hat_full[0]
        
        # 스무딩된 데이터로 DataFrame 생성
        smoothed_df = data.copy()
        smoothed_df[['deg1', 'deg2', 'deg3', 'deg4']] = smoothed_angles
        
        return smoothed_df

    def normalize_time(self, trajectory, num_points=None):
        """궤적을 시간에 대해 정규화"""
        if num_points is None:
            num_points = len(trajectory)
        current_length = len(trajectory)
        old_time = np.linspace(0, 1, current_length)
        new_time = np.linspace(0, 1, num_points)
        
        interpolated = np.zeros((num_points, trajectory.shape[1]))
        for i in range(trajectory.shape[1]):
            f = interp1d(old_time, trajectory[:, i], kind='cubic', bounds_error=False, fill_value="extrapolate")
            interpolated[:, i] = f(new_time)
        return interpolated

    # 궤적의 종류에 따라 다른 보간 방법 적용
    def interpolate_line(self, target, subject, interpolation_weight=0.5):
        """DTW 기반 선형 보간"""
        # 사용자 궤적 시간 정규화
        target_normalized = self.normalize_time(target, num_points=len(target))
        subject_normalized = self.normalize_time(subject, num_points=len(target))
        
        # DataFrame 생성
        target_df = pd.DataFrame(target_normalized, columns=['deg1', 'deg2', 'deg3', 'deg4'])
        subject_df = pd.DataFrame(subject_normalized, columns=['deg1', 'deg2', 'deg3', 'deg4'])

        # 데이터 스무딩 적용
        # target_smoothed = self.smooth_data(target_df)
        # subject_smoothed = self.smooth_data(subject_df)
        
        # DTW 거리 및 경로 계산
        distance, path = fastdtw(target_df, subject_df, dist=euclidean)
        path = np.array(path)

        # 매칭된 데이터 포인터 추출
        target_matched = target_normalized[path[:, 0]]
        subject_matched = subject_normalized[path[:, 1]]

        # 보간된 관절 각도 계산
        interpolated_degrees = np.zeros_like(target_matched)
        
        # 관절 각도 제한 적용
        for joint in range(4):
            current_diff = subject[:, joint] - target[:, joint]
            interpolated_degrees[:, joint] = target[:, joint] + current_diff * interpolation_weight
            
            # 관절 제한 적용
            min_val, max_val = self.joint_limits[joint]
            interpolated_degrees[:, joint] = np.clip(interpolated_degrees[:, joint], 
                                                min_val, max_val)
            
        return interpolated_degrees

    def interpolate_arc(self, target, subject):
        """Cubic Spline 기반 호 보간"""
        # 사용자 궤적 시간 정규화
        target_normalized = self.normalize_time(target, num_points=len(target))
        subject_normalized = self.normalize_time(subject, num_points=len(target))

        interpolated_degrees = np.zeros((len(target_normalized), 4))
        t = np.linspace(0, 1, len(target_normalized)) 
        
        # 각 관절별 다른 가중치 설정
        joint_weights = {
            0: 0.5,    
            1: 0.5,    
            2: 0.5,    
            3: 0.5     
        }
        
        # 각 관절별 보간
        for joint in range(4):
            target_interp = target_normalized[:, joint]
            subject_interp = subject_normalized[:, joint]
            
            # 각도 차이 계산 및 보정
            angle_diffs = subject_interp - target_interp
            angle_diffs = np.where(angle_diffs > 180, angle_diffs - 360,
                                np.where(angle_diffs < -180, angle_diffs + 360, angle_diffs))
            
            # 현재 관절의 가중치 가져오기
            weight = joint_weights[joint]
            
            # 제어점 설정 (시작점, 중간점, 끝점)
            control_points = np.array([
                target_interp[0],  # 시작점
                target_interp[0] + weight * angle_diffs[0],  # 중간 제어점 
                target_interp[-1] + weight * angle_diffs[-1],  # 중간 제어점 
                target_interp[-1] + angle_diffs[-1]  # 끝점
            ])
            
            # Cubic Spline 보간 적용
            cs = CubicSpline(np.linspace(0, 1, 4), control_points)
            interpolated_angles = cs(t)
            
            # 관절 제한 적용
            min_val, max_val = self.joint_limits[joint]
            interpolated_degrees[:, joint] = np.clip(interpolated_angles, min_val, max_val)
        
        return interpolated_degrees

    def interpolate_circle(self, target, subject):
        """SLERP 기반 원형 보간"""
        # 사용자 궤적 시간 정규화
        target_normalized = self.normalize_time(target, num_points=len(target))
        subject_normalized = self.normalize_time(subject, num_points=len(target))

        interpolated_degrees = np.zeros((len(target), 4))
        
        # 각 관절별 가중치 설정 c (조정 알고리즘 추가해서 적용)
        joint_weights = {
            0: np.linspace(0, 0.5, len(target)),  # Joint 1 (deg1)
            1: np.linspace(0, 0.5, len(target)), # Joint 2 (deg2)
            2: np.linspace(0, 0.5, len(target)), # Joint 3 (deg3)
            3: np.linspace(0, 0.5, len(target))  # Joint 4 (deg4)
        }
        
        for joint in range(4):
            target_joint = target_normalized[:, joint]
            subject_joint = subject_normalized[:, joint]

            # 현재 관절에 대한 가중치 가져오기
            t = joint_weights[joint]
            
            # 각도 차이 계산 및 보정
            angle_diff = subject_joint - target_joint
            angle_diff = np.where(angle_diff > 180, angle_diff - 360,
                                np.where(angle_diff < -180, angle_diff + 360, angle_diff))
            
            # SLERP 보간
            omega = np.abs(angle_diff)  # 회전 각도

            for i, weight in enumerate(t):
                if omega[i] < 1e-6:  # 각도 차이가 매우 작을 경우
                    interpolated_degrees[i, joint] = target_joint[i] + weight * angle_diff[i]
                else:
                    sin_omega = np.sin(np.radians(omega[i]))  # 여기서만 라디안 사용
                    interpolated_degrees[i, joint] = (
                        target_joint[i] * np.sin(np.radians((1-weight)*omega[i]))/sin_omega + 
                        subject_joint[i] * np.sin(np.radians(weight*omega[i]))/sin_omega
                    )
            
            # 관절 제한 적용
            min_val, max_val = self.joint_limits[joint]
            interpolated_degrees[:, joint] = np.clip(interpolated_degrees[:, joint], 
                                                min_val, max_val)
        
        return interpolated_degrees

    def interpolate_trajectory(self, target_df, user_df, trajectory_type):
        """궤적 타입에 따른 보간 수행"""
        # 관절 각도 데이터 추출
        target_degrees = target_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_degrees = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values

        # 궤적 타입 분류 및 적절한 보간 방법 선택
        main_type = self.analyzer.classify_trajectory_type(trajectory_type)
        if main_type == 'line':
            aligned_degrees = self.interpolate_line(target_degrees, user_degrees)
        elif main_type == 'arc':
            aligned_degrees = self.interpolate_arc(target_degrees, user_degrees)
        elif main_type == 'circle':
            aligned_degrees = self.interpolate_circle(target_degrees, user_degrees)
        else:
            raise ValueError(f"Unknown trajectory type: {main_type}")

        # 보간된 관절 각도로부터 end-effector 위치 계산
        aligned_points = np.array([calculate_end_effector_position(deg) for deg in aligned_degrees])
        aligned_points = aligned_points * 1000

        # end-effector 위치 범위 제한
        for i, col in enumerate(['x_end', 'y_end', 'z_end']):
            min_val = min(target_df[col].min(), user_df[col].min())
            max_val = max(target_df[col].max(), user_df[col].max())
            aligned_points[:, i] = np.clip(aligned_points[:, i], min_val, max_val)

        # 결과 데이터프레임 생성
        generated_df = pd.DataFrame(
            np.column_stack([aligned_points, aligned_degrees]),
            columns=['x_end', 'y_end', 'z_end', 'deg1', 'deg2', 'deg3', 'deg4']
        )
        
        return generated_df
    
    def visualize_trajectories(self, target_df, user_df, generated_df, trajectory_type, generation_number=1):
        """타겟과 사용자 궤적과 생성된 궤적을 시각화"""
        
        # 모든 데이터를 최소 길이로 맞춤
        # min_length = min(len(target_df), len(user_df), len(generated_df))
        # target_df = target_df.iloc[:min_length].copy()
        # user_df = user_df.iloc[:min_length].copy()  
        # generated_df = generated_df.iloc[:min_length].copy()
        
        # print(f"Generated: {generated_df.shape}")
        
        # # 관절 각도 데이터로 DTW와 보간 수행
        target_degrees = target_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_degrees = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        generated_degrees = generated_df[['deg1', 'deg2', 'deg3', 'deg4']].values

        # 디버깅용
        # print("\nData verification:")
        # print(f"Target shape: {target_df.shape}")
        # print(f"User shape: {user_df.shape}")
        # print(f"Generated shape: {generated_df.shape}")
        
        # 데이터 샘플 확인
        # print("\nFirst few rows of generated data:")
        # print(generated_df.head())
        
        # 3D 궤적 데이터 범위 확인
        print("\nRange of end-effector positions:")
        for df, name in [(target_df, 'Target'), (user_df, 'User'), (generated_df, 'Generated')]:
            print(f"\n{name}:")
            for col in ['x_end', 'y_end', 'z_end']:
                print(f"{col}: {df[col].min():.2f} to {df[col].max():.2f}")

        # 시각화
        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(2, 3, figure=fig)
        
        # 3D end-effector 궤적 시각화
        ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
        ax_3d.plot(target_df['x_end'], target_df['y_end'], target_df['z_end'], 
                'b-', label='Target')
        ax_3d.plot(user_df['x_end'], user_df['y_end'], user_df['z_end'],
                'r-', label='User')
        ax_3d.plot(generated_df['x_end'], generated_df['y_end'], generated_df['z_end'], 
                'g-', linewidth=2, label='Generated')
        
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title('End-Effector Trajectory')
        ax_3d.legend()

        # 관절 각도 그래프
        target_time = np.arange(len(target_degrees))
        user_time = np.arange(len(user_degrees))
        aligned_time = np.arange(len(generated_degrees))
        joint_titles = ['Joint 1(degree 1)', 'Joint 2(degree 2)', 'Joint 3(degree 3)', 'Joint 4(degree 4)']
        for idx, joint in enumerate(['deg1', 'deg2', 'deg3', 'deg4']):
            row = idx // 2
            col = (idx % 2) + 1
            
            ax = fig.add_subplot(gs[row, col])
            
            ax.plot(target_time, target_df[joint], 'b--', label='Target')
            ax.plot(user_time, user_df[joint], 'r:', label='User')
            ax.plot(aligned_time, generated_df[joint], 'g-', label='Generated')
            
            ax.set_title(joint_titles[idx])
            ax.set_xlabel('Time step')
            ax.set_ylabel('Angle (deg)')
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()

        # 생성된 궤적 저장
        self.save_generated_trajectory(generated_df, trajectory_type, generation_number)
        
        return generated_df
    
    def save_generated_trajectory(self, generated_df, classification_result, generation_number=1):
        """생성된 궤적을 지정된 형식으로 저장"""
        generation_dir = os.path.join(os.getcwd(), "generation_trajectory")
        os.makedirs(generation_dir, exist_ok=True)
        
        filename = f"generation_trajectory_{classification_result}_{generation_number}.txt"
        generation_path = os.path.join(generation_dir, filename)
        
        # 데이터프레임 생성 및 형식 맞추기
        num_points = len(generated_df)
        full_df = pd.DataFrame(index=range(num_points))
        
        # 기본 필드 설정
        full_df['r'] = 'm'
        full_df['sequence'] = range(num_points)
        full_df['timestamp'] = [i * 20 for i in range(num_points)]  # 20ms 간격
        
        # 각도, 엔드이펙터터의 데이터 설정
        full_df['deg'] = (generated_df['deg1'].round(3).astype(str) + '/' + 
                        generated_df['deg2'].round(3).astype(str) + '/' +
                        generated_df['deg3'].round(3).astype(str) + '/' +
                        generated_df['deg4'].round(3).astype(str))

        full_df['endpoint'] = (generated_df['x_end'].round(3).astype(str) + '/' + 
                            generated_df['y_end'].round(3).astype(str) + '/' + 
                            generated_df['z_end'].round(3).astype(str))
        
        # 열 순서 지정 및 저장
        columns = ['deg', 'endpoint']
        full_df = full_df[columns]
        
        full_df.to_csv(generation_path, index=False)
        print(f"\nGenerated trajectory saved to: {generation_path}")

        return generation_path

def main():
    base_dir = os.path.join(os.getcwd(), "data")
    
    try:
        # 객체 초기화
        analyzer = TrajectoryAnalyzer(
            classification_model="best_classification_model.pth",
            base_dir=base_dir
        )
        generator = TrajectoryGenerator(analyzer)

        # 사용자 궤적 파일 로드
        print("\nLoading and classifying user trajectories...")
        non_golden_dir = os.path.join(base_dir, "non_golden_sample")
        non_golden_files = [f for f in os.listdir(non_golden_dir) if f.endswith('.txt')]
        
        if not non_golden_files:
            raise ValueError("No trajectory files found in the non_golden_sample directory.")
        
        selected_file = random.choice(non_golden_files)
        print(f"Selected user trajectory: {selected_file}")
        
        file_path = os.path.join(non_golden_dir, selected_file)

        # 궤적 로드 및 분류
        user_trajectory, trajectory_type = analyzer.load_user_trajectory(file_path)
        target_trajectory, _ = analyzer.load_target_trajectory(trajectory_type)
        
        # 궤적 생성
        generated_df = generator.interpolate_trajectory(
            target_df=target_trajectory,
            user_df=user_trajectory,
            trajectory_type=trajectory_type
        )

        # 시각화 및 저장
        print("\nVisualizing and saving trajectories...")
        generator.visualize_trajectories(
            target_df=target_trajectory,
            user_df=user_trajectory,
            generated_df=generated_df,
            trajectory_type=trajectory_type
        )
        
        print("\nProcessing completed!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Please check the data directory structure and model path.")

if __name__ == "__main__":
    main()

   