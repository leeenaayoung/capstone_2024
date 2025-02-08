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

class TrajectoryGenerator:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.joint_limits = {
            0: (-10, 110),   # degree 제한 범위 적용
            1: (0, 150),     
            2: (0, 150),     
            3: (-90, 90)     
        }

    def calculate_time_intervals(self, degrees, velocities):
        """각도 변화와 속도를 기반으로 시간 간격 계산"""
        time_intervals = np.zeros(len(degrees))
        
        for i in range(1, len(degrees)):
            degree_changes = np.abs(degrees[i] - degrees[i-1])
            max_velocities = np.maximum(np.abs(velocities[i]), np.abs(velocities[i-1]))
            max_velocities = np.where(max_velocities < 1e-6, 1e-6, max_velocities)
            required_times = degree_changes / max_velocities
            time_intervals[i] = np.max(required_times)
            
        return time_intervals
    
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


    def interpolate_line(self, target, subject, target_vel, subject_vel, interpolation_weight=0.5):
        """DTW 기반 선형 보간 (속도 고려)"""
        # 시간 정규화
        target_normalized = self.normalize_time(target, num_points=len(target))
        subject_normalized = self.normalize_time(subject, num_points=len(target))
        target_vel_normalized = self.normalize_time(target_vel, num_points=len(target))
        subject_vel_normalized = self.normalize_time(subject_vel, num_points=len(target))
        
        # DTW 거리 및 경로 계산
        distance, path = fastdtw(target_normalized, subject_normalized, dist=euclidean)
        path = np.array(path)
        
        # 보간된 값을 저장할 배열 초기화
        interpolated_degrees = np.zeros((len(path), 4))
        interpolated_velocities = np.zeros((len(path), 4))
        
        # 각 시점에서 보간 수행
        for i, (target_idx, subject_idx) in enumerate(path):
            for joint in range(4):
                # 각도와 속도 보간
                interpolated_degrees[i, joint] = (1 - interpolation_weight) * target_normalized[target_idx, joint] + \
                                               interpolation_weight * subject_normalized[subject_idx, joint]
                interpolated_velocities[i, joint] = (1 - interpolation_weight) * target_vel_normalized[target_idx, joint] + \
                                                   interpolation_weight * subject_vel_normalized[subject_idx, joint]
                
                # 관절 제한 적용
                min_val, max_val = self.joint_limits[joint]
                interpolated_degrees[i, joint] = np.clip(interpolated_degrees[i, joint], min_val, max_val)
        
        # 시간 간격 계산
        time_intervals = self.calculate_time_intervals(interpolated_degrees, interpolated_velocities)
        
        return interpolated_degrees, interpolated_velocities, time_intervals

    def interpolate_arc(self, target, subject, target_vel, subject_vel):
        """Cubic Spline 기반 호 보간 (속도 고려)"""
        # 시간 정규화
        target_normalized = self.normalize_time(target, num_points=len(target))
        subject_normalized = self.normalize_time(subject, num_points=len(target))
        target_vel_normalized = self.normalize_time(target_vel, num_points=len(target))
        subject_vel_normalized = self.normalize_time(subject_vel, num_points=len(target))

        interpolated_degrees = np.zeros((len(target_normalized), 4))
        interpolated_velocities = np.zeros((len(target_normalized), 4))
        t = np.linspace(0, 1, len(target_normalized))
        
        for joint in range(4):
            # 각도에 대한 Cubic Spline
            control_points_deg = np.array([
                target_normalized[0, joint],
                target_normalized[len(target_normalized)//3, joint],
                subject_normalized[2*len(subject_normalized)//3, joint],
                subject_normalized[-1, joint]
            ])
            cs_deg = CubicSpline(np.linspace(0, 1, 4), control_points_deg)
            
            # 속도에 대한 Cubic Spline
            control_points_vel = np.array([
                target_vel_normalized[0, joint],
                target_vel_normalized[len(target_vel_normalized)//3, joint],
                subject_vel_normalized[2*len(subject_vel_normalized)//3, joint],
                subject_vel_normalized[-1, joint]
            ])
            cs_vel = CubicSpline(np.linspace(0, 1, 4), control_points_vel)
            
            # 보간 적용
            interpolated_degrees[:, joint] = cs_deg(t)
            interpolated_velocities[:, joint] = cs_vel(t)
            
            # 관절 제한 적용
            min_val, max_val = self.joint_limits[joint]
            interpolated_degrees[:, joint] = np.clip(interpolated_degrees[:, joint], min_val, max_val)
        
        # 시간 간격 계산
        time_intervals = self.calculate_time_intervals(interpolated_degrees, interpolated_velocities)
        
        return interpolated_degrees, interpolated_velocities, time_intervals

    def interpolate_circle(self, target, subject, target_vel, subject_vel):
        """SLERP 기반 원형 보간 (속도 고려)"""
        # 시간 정규화
        target_normalized = self.normalize_time(target, num_points=len(target))
        subject_normalized = self.normalize_time(subject, num_points=len(target))
        target_vel_normalized = self.normalize_time(target_vel, num_points=len(target))
        subject_vel_normalized = self.normalize_time(subject_vel, num_points=len(target))

        interpolated_degrees = np.zeros((len(target), 4))
        interpolated_velocities = np.zeros((len(target), 4))
        
        # 각 관절별 가중치 설정
        t = np.linspace(0, 0.5, len(target))
        
        for joint in range(4):
            target_joint = target_normalized[:, joint]
            subject_joint = subject_normalized[:, joint]
            target_vel_joint = target_vel_normalized[:, joint]
            subject_vel_joint = subject_vel_normalized[:, joint]

            # 각도 차이 계산 및 보정
            angle_diff = subject_joint - target_joint
            angle_diff = np.where(angle_diff > 180, angle_diff - 360,
                                np.where(angle_diff < -180, angle_diff + 360, angle_diff))
            
            # 속도 차이 계산
            vel_diff = subject_vel_joint - target_vel_joint
            
            # SLERP 보간
            omega = np.abs(angle_diff)
            
            for i, weight in enumerate(t):
                if omega[i] < 1e-6:
                    interpolated_degrees[i, joint] = target_joint[i] + weight * angle_diff[i]
                    interpolated_velocities[i, joint] = target_vel_joint[i] + weight * vel_diff[i]
                else:
                    sin_omega = np.sin(np.radians(omega[i]))
                    interpolated_degrees[i, joint] = (
                        target_joint[i] * np.sin(np.radians((1-weight)*omega[i]))/sin_omega + 
                        subject_joint[i] * np.sin(np.radians(weight*omega[i]))/sin_omega
                    )
                    interpolated_velocities[i, joint] = (
                        target_vel_joint[i] * (1-weight) + subject_vel_joint[i] * weight
                    )
            
            # 관절 제한 적용
            min_val, max_val = self.joint_limits[joint]
            interpolated_degrees[:, joint] = np.clip(interpolated_degrees[:, joint], min_val, max_val)
        
        # 시간 간격 계산
        time_intervals = self.calculate_time_intervals(interpolated_degrees, interpolated_velocities)
        
        return interpolated_degrees, interpolated_velocities, time_intervals

    def interpolate_trajectory(self, target_df, user_df, trajectory_type):
        """궤적 타입에 따른 보간 수행"""
        # 각도와 속도 데이터 추출
        if 'degsec1' not in target_df.columns or 'degsec1' not in user_df.columns:
            raise ValueError("Velocity data (degsec1-4) not found in input dataframes")
            
        target_degrees = target_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        target_velocities = target_df[['degsec1', 'degsec2', 'degsec3', 'degsec4']].values
        user_degrees = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_velocities = user_df[['degsec1', 'degsec2', 'degsec3', 'degsec4']].values

        # 궤적 타입에 따른 보간 방법 선택
        main_type = self.analyzer.classify_trajectory_type(trajectory_type)
        if main_type == 'line':
            degrees, velocities, time_intervals = self.interpolate_line(
                target_degrees, user_degrees, target_velocities, user_velocities)
        elif main_type == 'arc':
            degrees, velocities, time_intervals = self.interpolate_arc(
                target_degrees, user_degrees, target_velocities, user_velocities)
        elif main_type == 'circle':
            degrees, velocities, time_intervals = self.interpolate_circle(
                target_degrees, user_degrees, target_velocities, user_velocities)
        else:
            raise ValueError(f"Unknown trajectory type: {main_type}")

        # 시간 계산
        cumulative_time = np.cumsum(time_intervals) * 1000  # 밀리초 단위로 변환
        
        # end-effector 위치 계산
        points = np.array([calculate_end_effector_position(deg) for deg in degrees])
        points = points * 1000

        # 결과 데이터프레임 생성
        generated_df = pd.DataFrame({
            'x_end': points[:, 0],
            'y_end': points[:, 1],
            'z_end': points[:, 2],
            'deg1': degrees[:, 0],
            'deg2': degrees[:, 1],
            'deg3': degrees[:, 2],
            'deg4': degrees[:, 3],
            'degsec1': velocities[:, 0],
            'degsec2': velocities[:, 1],
            'degsec3': velocities[:, 2],
            'degsec4': velocities[:, 3],
            'timestamp': cumulative_time
        })
        
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
    
    def visualize_trajectories(self, target_df, user_df, generated_df, trajectory_type, generation_number=1):
        """타겟과 사용자 궤적과 생성된 궤적을 시각화 (속도 포함)"""
        # 데이터 범위 확인
        print("\nRange of end-effector positions:")
        for df, name in [(target_df, 'Target'), (user_df, 'User'), (generated_df, 'Generated')]:
            print(f"\n{name}:")
            for col in ['x_end', 'y_end', 'z_end']:
                print(f"{col}: {df[col].min():.2f} to {df[col].max():.2f}")

        # 3x3 그리드로 시각화 (궤적 + 각도 + 속도)
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 3, figure=fig)
        
        # 3D end-effector 궤적 시각화
        ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
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

        # 시간 데이터 준비
        target_time = target_df['timestamp'] if 'timestamp' in target_df else np.arange(len(target_df))
        user_time = user_df['timestamp'] if 'timestamp' in user_df else np.arange(len(user_df))
        generated_time = generated_df['timestamp']

        # 관절 각도 그래프
        joint_titles = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4']
        for idx, (joint, vel) in enumerate(zip(['deg1', 'deg2', 'deg3', 'deg4'], 
                                             ['degsec1', 'degsec2', 'degsec3', 'degsec4'])):
            # 각도 그래프
            ax_deg = fig.add_subplot(gs[1, idx])
            ax_deg.plot(target_time, target_df[joint], 'b--', label='Target')
            ax_deg.plot(user_time, user_df[joint], 'r:', label='User')
            ax_deg.plot(generated_time, generated_df[joint], 'g-', label='Generated')
            ax_deg.set_title(f'{joint_titles[idx]} Angle')
            ax_deg.set_xlabel('Time (ms)')
            ax_deg.set_ylabel('Angle (deg)')
            ax_deg.grid(True)
            ax_deg.legend()

            # 속도 그래프
            ax_vel = fig.add_subplot(gs[2, idx])
            ax_vel.plot(target_time, target_df[vel], 'b--', label='Target')
            ax_vel.plot(user_time, user_df[vel], 'r:', label='User')
            ax_vel.plot(generated_time, generated_df[vel], 'g-', label='Generated')
            ax_vel.set_title(f'{joint_titles[idx]} Velocity')
            ax_vel.set_xlabel('Time (ms)')
            ax_vel.set_ylabel('Velocity (deg/s)')
            ax_vel.grid(True)
            ax_vel.legend()

        plt.tight_layout()
        plt.show()

        # 생성된 궤적 저장
        self.save_generated_trajectory(generated_df, trajectory_type, generation_number)
        
        return generated_df

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
        
        # 속도 데이터 확인
        required_columns = ['deg1', 'deg2', 'deg3', 'deg4', 'degsec1', 'degsec2', 'degsec3', 'degsec4']
        if not all(col in user_trajectory.columns for col in required_columns) or \
           not all(col in target_trajectory.columns for col in required_columns):
            raise ValueError("Missing required columns (degrees or velocities) in trajectory data")
        
        # 궤적 생성
        print(f"\nGenerating trajectory for type: {trajectory_type}")
        generated_df = generator.interpolate_trajectory(
            target_df=target_trajectory,
            user_df=user_trajectory,
            trajectory_type=trajectory_type
        )

        # 시각화 및 저장
        print("\nVisualizing trajectories...")
        generator.visualize_trajectories(
            target_df=target_trajectory,
            user_df=user_trajectory,
            generated_df=generated_df,
            trajectory_type=trajectory_type
        )
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Please check the data directory structure and required columns in the input files.")

if __name__ == "__main__":
    main()