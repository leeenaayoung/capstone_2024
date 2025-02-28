import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from fastdtw import fastdtw
from scipy.interpolate import CubicSpline
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
            0: (-10, 110),
            1: (0, 150),     
            2: (0, 150),     
            3: (-90, 90)     
        }

    def smooth_data(self, data, R=0.02, Q=0.1):
        """칼만 필터를 사용한 데이터 스무딩"""
        # 각도와 각속도 데이터 추출
        angles = data[['deg1', 'deg2', 'deg3', 'deg4']].values
        velocities = data[['degsec1', 'degsec2', 'degsec3', 'degsec4']].values
        n_samples, n_joints = angles.shape
        
        # 결과를 저장할 배열 초기화
        smoothed_angles = np.zeros_like(angles)
        smoothed_velocities = np.zeros_like(velocities)
        
        for joint in range(n_joints):   
            # 초기 상태 설정
            x_hat_full = np.array([angles[0, joint], velocities[0, joint]])
            P_full = np.eye(2)
            
            # 시스템 행렬 설정
            dt = 1.0
            A = np.array([[1, dt],
                        [0, 1]])
            H = np.eye(2) 
            
            # 프로세스 노이즈 설정
            Q_matrix = Q * np.array([[dt**4/4, dt**3/2],
                                    [dt**3/2, dt**2]])
            # 측정 노이즈 설정
            R_matrix = np.diag([R, R*10]) 
            
            # 첫 번째 상태 저장
            smoothed_angles[0, joint] = x_hat_full[0]
            smoothed_velocities[0, joint] = x_hat_full[1]
            
            # 칼만 필터 적용
            for k in range(1, n_samples):
                # 예측 단계
                x_hat_full = A @ x_hat_full
                P_full = A @ P_full @ A.T + Q_matrix
                
                # 현재 측정값
                z = np.array([angles[k, joint], velocities[k, joint]])
                
                # 업데이트 단계
                y = z - H @ x_hat_full
                S = H @ P_full @ H.T + R_matrix
                K = P_full @ H.T @ np.linalg.inv(S)
                
                # 상태 업데이트
                x_hat_full = x_hat_full + K @ y
                P_full = (np.eye(2) - K @ H) @ P_full
                
                # 결과 저장
                smoothed_angles[k, joint] = x_hat_full[0]
                smoothed_velocities[k, joint] = x_hat_full[1]
        
        # 결과를 데이터프레임으로 변환
        smoothed_df = data.copy()
        smoothed_df[['deg1', 'deg2', 'deg3', 'deg4']] = smoothed_angles
        smoothed_df[['degsec1', 'degsec2', 'degsec3', 'degsec4']] = smoothed_velocities
        
        return smoothed_df

    def normalize_time(self, target_trajectory, subject_trajectory):
        """궤적을 시간에 대해 정규화"""
        target_angles = target_trajectory[:, :4]
        subject_angles = subject_trajectory[:, :4]
        
        # 각도 데이터 정규화 (joint limits 기반)
        normalized_target_angles = np.zeros_like(target_angles)
        normalized_subject_angles = np.zeros_like(subject_angles)
        
        for joint in range(4):
            min_val = self.joint_limits[joint][0] 
            max_val = self.joint_limits[joint][1] 
            range_val = max_val - min_val
            normalized_target_angles[:, joint] = (target_angles[:, joint] - min_val) / range_val
            normalized_subject_angles[:, joint] = (subject_angles[:, joint] - min_val) / range_val
        
        # DTW로 정렬
        _, path = fastdtw(normalized_target_angles, normalized_subject_angles, dist=euclidean)
        path = np.array(path, dtype=np.int32)
        
        # 찾은 경로로 전체 궤적 정렬 (각도 + 각속도)
        aligned_target = target_trajectory[path[:, 0]]
        aligned_subject = subject_trajectory[path[:, 1]]
        
        return aligned_target, aligned_subject
        
    # 궤적의 종류에 따라 다른 보간 방법 적용
    def interpolate_line(self, target, subject, interpolation_weight=0.5):
        """DTW 기반 선형 보간"""
        # 데이터 분리
        target_angles = target[:, :4]  
        target_velocities = target[:, 4:]
        subject_angles = subject[:, :4]
        subject_velocities = subject[:, 4:]

        # DTW를 통한 시간 정렬
        aligned_target, aligned_subject = self.normalize_time(target, subject)
        
        # 정렬된 데이터 분리
        aligned_target_angles = aligned_target[:, :4]
        aligned_target_velocities = aligned_target[:, 4:]
        aligned_subject_angles = aligned_subject[:, :4]
        aligned_subject_velocities = aligned_subject[:, 4:]

        # 결과 저장을 위한 배열 초기화
        interpolated_degrees = np.zeros_like(aligned_target_angles)
        interpolated_velocities = np.zeros_like(aligned_target_velocities)

        for joint in range(4):
            # 현재 joint의 각도 범위 계산
            target_range = np.ptp(aligned_target_angles[:, joint])
            subject_range = np.ptp(aligned_subject_angles[:, joint])
            
            # 보간 가중치 동적 조정
            local_weight = interpolation_weight
            if target_range > subject_range:
                # 타겟의 움직임이 더 큰 경우, 타겟 쪽으로 더 치우친 보간
                local_weight = interpolation_weight * 0.8
            elif subject_range > target_range:
                # 사용자의 움직임이 더 큰 경우, 사용자 쪽으로 더 치우친 보간
                local_weight = interpolation_weight * 1.2

            # 각도 차이 계산 및 보간
            angle_diff = aligned_subject_angles[:, joint] - aligned_target_angles[:, joint]
            base_interpolation = aligned_target_angles[:, joint] + angle_diff * local_weight

            # 각속도 고려한 보정
            velocity_diff = aligned_subject_velocities[:, joint] - aligned_target_velocities[:, joint]
            velocity_correction = velocity_diff * local_weight * 0.1  # 각속도의 영향력 조절
            
            # 최종 각도 계산
            interpolated_degrees[:, joint] = base_interpolation + velocity_correction

            # 각속도 보간
            interpolated_velocities[:, joint] = (
                aligned_target_velocities[:, joint] + velocity_diff * local_weight
            )

            # 관절 제한 부드럽게 적용
            min_val, max_val = self.joint_limits[joint]
            interpolated_degrees[:, joint] = np.clip(
                interpolated_degrees[:, joint], min_val, max_val
            )

        return interpolated_degrees, interpolated_velocities
    
    def interpolate_arc(self, target, subject):
        """Cubic Spline 기반 호 보간"""
        # 시간 정규화
        aligned_target, aligned_subject = self.normalize_time(target, subject)
        
        # 데이터 분리
        aligned_target_angles = aligned_target[:, :4]
        aligned_target_velocities = aligned_target[:, 4:]
        aligned_subject_angles = aligned_subject[:, :4]
        aligned_subject_velocities = aligned_subject[:, 4:]

        n_points = len(aligned_target_angles)
        interpolated_degrees = np.zeros_like(aligned_target_angles)
        interpolated_velocities = np.zeros_like(aligned_target_velocities)
        
        # 시간 포인트
        t = np.linspace(0, 1, n_points)
        
        for joint in range(4):
            # 3개의 제어점 사용
            control_indices = [0, n_points//2, -1]
            control_times = [0, 0.5, 1]
            
            # 제어점의 각도와 각속도 계산
            control_angles = np.zeros(len(control_indices))
            control_velocities = np.zeros(len(control_indices))
            
            for i, idx in enumerate(control_indices):
                control_angles[i] = (0.6 * aligned_target_angles[idx, joint] + 
                                    0.4 * aligned_subject_angles[idx, joint])
                control_velocities[i] = (0.6 * aligned_target_velocities[idx, joint] + 
                                        0.4 * aligned_subject_velocities[idx, joint])
            
            # Cubic Spline 보간
            cs_angle = CubicSpline(control_times, control_angles)
            cs_vel = CubicSpline(control_times, control_velocities)
            
            # 보간된 값 계산
            interpolated_degrees[:, joint] = cs_angle(t)
            interpolated_velocities[:, joint] = cs_vel(t)
            
            # 관절 제한 적용
            min_val, max_val = self.joint_limits[joint]
            interpolated_degrees[:, joint] = np.clip(interpolated_degrees[:, joint], min_val, max_val)

        return interpolated_degrees, interpolated_velocities

    # 쿼터니안 적용 x 보간 코드
    # def interpolate_circle(self, target, subject):
    #         """원형 운동을 위한 SLERP 보간"""
    #         aligned_target, aligned_subject = self.normalize_time(target, subject)

    #         aligned_target_angles = aligned_target[:, :4]
    #         aligned_target_velocities = aligned_target[:, 4:]
    #         aligned_subject_angles = aligned_subject[:, :4]
    #         aligned_subject_velocities = aligned_subject[:, 4:]

    #         n_points = len(aligned_target_angles)
    #         interpolated_degrees = np.zeros_like(aligned_target_angles)
    #         interpolated_velocities = np.zeros_like(aligned_target_velocities)

    #         t = np.linspace(0, 1, n_points)

    #         # 원형 운동을 위한 사인 기반 가중치
    #         phase = 2 * np.pi * t
    #         weights = (1 - np.cos(phase)) / 2

    #         for joint in range(4):
    #             # 각도 차이 계산 (360도 고려)
    #             angle_diff = aligned_subject_angles[:, joint] - aligned_target_angles[:, joint]
    #             angle_diff = np.where(angle_diff > 180, angle_diff - 360,
    #                                 np.where(angle_diff < -180, angle_diff + 360, angle_diff))

    #             # SLERP로 보간
    #             for i, w in enumerate(weights):
    #                 theta = np.radians(angle_diff[i])
    #                 if np.abs(theta) < 1e-6:
    #                     interpolated_degrees[i, joint] = aligned_target_angles[i, joint]
    #                 else:
    #                     sin_theta = np.sin(theta)
    #                     interpolated_degrees[i, joint] = (
    #                         aligned_target_angles[i, joint] * np.sin((1-w) * theta) / sin_theta +
    #                         aligned_subject_angles[i, joint] * np.sin(w * theta) / sin_theta
    #                     )

    #                 # 각속도는 선형 보간
    #                 interpolated_velocities[i, joint] = (
    #                     (1-w) * aligned_target_velocities[i, joint] +
    #                     w * aligned_subject_velocities[i, joint]
    #                 )

    #             # 관절 제한 적용
    #             min_val, max_val = self.joint_limits[joint]
    #             interpolated_degrees[:, joint] = np.clip(interpolated_degrees[:, joint], min_val, max_val)

    #         return interpolated_degrees, interpolated_velocities

    # 쿼터니안 적용
    def interpolate_circle(self, target, subject):
        """쿼터니언 기반 SLERP를 사용한 원형 운동 보간"""
        aligned_target, aligned_subject = self.normalize_time(target, subject)
        
        aligned_target_angles = aligned_target[:, :4]
        aligned_target_velocities = aligned_target[:, 4:]
        aligned_subject_angles = aligned_subject[:, :4]
        aligned_subject_velocities = aligned_subject[:, 4:]

        n_points = len(aligned_target_angles)
        interpolated_degrees = np.zeros_like(aligned_target_angles)
        interpolated_velocities = np.zeros_like(aligned_target_velocities)

        t = np.linspace(0, 1, n_points)
        weights = t * t * (3 - 2 * t)  # smooth step function

        from scipy.spatial.transform import Rotation as R
        
        for i in range(n_points):
            # 타겟과 서브젝트의 각도를 라디안으로 변환
            target_rad = np.radians(aligned_target_angles[i])
            subject_rad = np.radians(aligned_subject_angles[i])
            
            # 각도를 회전 객체로 변환
            q_target = R.from_euler('xyz', target_rad[:3])
            q_subject = R.from_euler('xyz', subject_rad[:3])
            
            # 쿼터니언 값 추출
            q_target_arr = q_target.as_quat()
            q_subject_arr = q_subject.as_quat()
            
            # SLERP 직접 구현
            dot = np.sum(q_target_arr * q_subject_arr)
            
            # 최단 경로 보장
            if dot < 0:
                q_subject_arr = -q_subject_arr
                dot = -dot
                
            # 각도가 매우 작은 경우 선형 보간
            if dot > 0.9995:
                result = q_target_arr + weights[i] * (q_subject_arr - q_target_arr)
                result = result / np.linalg.norm(result)
            else:
                theta_0 = np.arccos(dot)
                sin_theta_0 = np.sin(theta_0)
                
                theta = theta_0 * weights[i]
                sin_theta = np.sin(theta)
                
                s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
                s1 = sin_theta / sin_theta_0
                
                result = s0 * q_target_arr + s1 * q_subject_arr
                
            # 보간된 쿼터니언을 회전 객체로 변환
            q_interp = R.from_quat(result)
            
            # 오일러 각도로 변환
            euler_angles = q_interp.as_euler('xyz', degrees=True)
            interpolated_degrees[i, :3] = euler_angles
            
            # 4번째 조인트는 선형 보간
            interpolated_degrees[i, 3] = (1 - weights[i]) * aligned_target_angles[i, 3] + \
                                    weights[i] * aligned_subject_angles[i, 3]
            
            # 각속도 보간
            for j in range(4):
                v0 = aligned_target_velocities[i, j]
                v1 = aligned_subject_velocities[i, j]
                w = weights[i]
                
                # Hermite 보간
                h00 = 2*w**3 - 3*w**2 + 1
                h10 = w**3 - 2*w**2 + w
                h01 = -2*w**3 + 3*w**2
                h11 = w**3 - w**2
                
                interpolated_velocities[i, j] = h00*v0 + h10*0 + h01*v1 + h11*0

        # 관절 제한 적용
        for joint in range(4):
            min_val, max_val = self.joint_limits[joint]
            
            # Smooth clamping using sigmoid
            def smooth_clamp(x, min_val, max_val):
                k = 10  # 시그모이드 기울기 계수
                x_normalized = (x - min_val) / (max_val - min_val)
                y_normalized = 1 / (1 + np.exp(-k * (x_normalized - 0.5)))
                return min_val + y_normalized * (max_val - min_val)
            
            interpolated_degrees[:, joint] = smooth_clamp(
                interpolated_degrees[:, joint], min_val, max_val)

        return interpolated_degrees, interpolated_velocities

    def interpolate_trajectory(self, target_df, user_df, trajectory_type):
        """궤적 타입에 따른 보간 수행"""        
        # 각속도 계산 및 DataFrame 생성
        target_with_vel = target_df.copy()
        user_with_vel = user_df.copy()
        
        # 각속도 계산 추가
        for df in [target_with_vel, user_with_vel]:
            df['degsec1'] = np.gradient(df['deg1'])
            df['degsec2'] = np.gradient(df['deg2'])
            df['degsec3'] = np.gradient(df['deg3'])
            df['degsec4'] = np.gradient(df['deg4'])

        # print("Velocities calculated")
        
        # 스무딩 적용
        target_smoothed = self.smooth_data(target_with_vel)
        user_smoothed = self.smooth_data(user_with_vel)

        # 관절 각도와 각속도 데이터 준비
        target_angles = target_smoothed[['deg1', 'deg2', 'deg3', 'deg4']].values
        target_velocities = target_smoothed[['degsec1', 'degsec2', 'degsec3', 'degsec4']].values
        
        user_angles = user_smoothed[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_velocities = user_smoothed[['degsec1', 'degsec2', 'degsec3', 'degsec4']].values
        
        target_data = np.column_stack([target_angles, target_velocities])
        user_data = np.column_stack([user_angles, user_velocities])
        
        # 보간 방법 선택 및 적용
        if 'clock' or 'counter' in trajectory_type.lower():
            print("Using circle interpolation")
            aligned_degrees, aligned_velocities = self.interpolate_circle(target_data, user_data)
        elif 'v_' or 'h_' in trajectory_type.lower():
            print("Using arc interpolation")
            aligned_degrees, aligned_velocities = self.interpolate_arc(target_data, user_data)
        else:
            print("Using line interpolation")
            aligned_degrees, aligned_velocities = self.interpolate_line(target_data, user_data)

        # print("Interpolation completed")
        
        # 보간된 관절 각도로부터 end-effector 위치 계산
        endeffector_degrees  = aligned_degrees.copy()
        endeffector_degrees[:, 1] -= 90
        endeffector_degrees[:, 3] -= 90

        aligned_points = np.array([calculate_end_effector_position(deg) for deg in endeffector_degrees])
        aligned_points = aligned_points * 1000

        # 결과 데이터프레임 생성
        generated_df = pd.DataFrame(
            np.column_stack([aligned_points, aligned_degrees]),
            columns=['x_end', 'y_end', 'z_end', 'deg1', 'deg2', 'deg3', 'deg4']
        )
        return generated_df

    def visualize_trajectories(self, target_df, user_df, generated_df, trajectory_type, generation_number=1):
        """타겟과 사용자 궤적과 생성된 궤적을 시각화"""
        # 관절 각도 데이터로 DTW와 보간 수행
        target_degrees = target_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_degrees = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        generated_degrees = generated_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        target_ends = np.array([calculate_end_effector_position(deg) for deg in target_degrees]) * 1000
        user_ends = np.array([calculate_end_effector_position(deg) for deg in user_degrees]) * 1000

        # 시각화
        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(2, 3, figure=fig)
        
        # 3D end-effector 궤적 시각화
        ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
        ax_3d.plot(target_ends[:, 0], target_ends[:, 1], target_ends[:, 2], 
                'b-', label='Target')
        ax_3d.plot(user_ends[:, 0], user_ends[:, 1], user_ends[:, 2],
                'r-', label='User')
        ax_3d.plot(generated_df['x_end'], generated_df['y_end'], generated_df['z_end'], 
                'g-', label='Generated')
        
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title('End-Effector Trajectory')
        ax_3d.legend()

        # joint 그래프
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
        full_df['timestamp'] = [i * 10 for i in range(num_points)]  # 20ms 간격
        
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