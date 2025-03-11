import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from fastdtw import fastdtw
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import euclidean
from torch import nn
from analyzer import TrajectoryAnalyzer
from utils import calculate_end_effector_position
from generation_m_model import JointTrajectoryTransformer
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

class ModelBasedTrajectoryGenerator:
    """모델 기반 궤적 생성기 클래스"""
    def __init__(self, analyzer, model_path=None):
        self.analyzer = analyzer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 관절 상관 관계 모델 초기화
        self.model = JointTrajectoryTransformer().to(self.device)
        
        # 모델 로드 (모델 경로가 제공된 경우)
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                state_dict = checkpoint['model_state_dict']
                # 현재 모델과 state_dict의 키를 비교하여 호환성 확인
                model_dict = self.model.state_dict()
                # 불필요한 키 제거
                state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                model_dict.update(state_dict)
                self.model.load_state_dict(model_dict)
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
        
        self.joint_limits = {0: (-10, 110), 1: (0, 150), 2: (0, 150), 3: (-90, 90)}
        self.model.eval()
        
    # def smooth_data(self, data, R=0.02, Q=0.1):
    #     """칼만 필터를 사용한 데이터 스무딩"""
    #     # 각도와 각속도 데이터 추출
    #     angles = data[['deg1', 'deg2', 'deg3', 'deg4']].values
    #     velocities = data[['degsec1', 'degsec2', 'degsec3', 'degsec4']].values
    #     n_samples, n_joints = angles.shape
        
    #     # 결과를 저장할 배열 초기화
    #     smoothed_angles = np.zeros_like(angles)
    #     smoothed_velocities = np.zeros_like(velocities)
        
    #     for joint in range(n_joints):   
    #         # 초기 상태 설정
    #         x_hat_full = np.array([angles[0, joint], velocities[0, joint]])
    #         P_full = np.eye(2)
            
    #         # 시스템 행렬 설정
    #         dt = 1.0
    #         A = np.array([[1, dt],
    #                     [0, 1]])
    #         H = np.eye(2) 
            
    #         # 프로세스 노이즈 설정
    #         Q_matrix = Q * np.array([[dt**4/4, dt**3/2],
    #                                 [dt**3/2, dt**2]])
    #         # 측정 노이즈 설정
    #         R_matrix = np.diag([R, R*10]) 
            
    #         # 첫 번째 상태 저장
    #         smoothed_angles[0, joint] = x_hat_full[0]
    #         smoothed_velocities[0, joint] = x_hat_full[1]
            
    #         # 칼만 필터 적용
    #         for k in range(1, n_samples):
    #             # 예측 단계
    #             x_hat_full = A @ x_hat_full
    #             P_full = A @ P_full @ A.T + Q_matrix
                
    #             # 현재 측정값
    #             z = np.array([angles[k, joint], velocities[k, joint]])
                
    #             # 업데이트 단계
    #             y = z - H @ x_hat_full
    #             S = H @ P_full @ H.T + R_matrix
    #             K = P_full @ H.T @ np.linalg.inv(S)
                
    #             # 상태 업데이트
    #             x_hat_full = x_hat_full + K @ y
    #             P_full = (np.eye(2) - K @ H) @ P_full
                
    #             # 결과 저장
    #             smoothed_angles[k, joint] = x_hat_full[0]
    #             smoothed_velocities[k, joint] = x_hat_full[1]
        
    #     # 결과를 데이터프레임으로 변환
    #     smoothed_df = data.copy()
    #     smoothed_df[['deg1', 'deg2', 'deg3', 'deg4']] = smoothed_angles
    #     smoothed_df[['degsec1', 'degsec2', 'degsec3', 'degsec4']] = smoothed_velocities
        
    #     return smoothed_df

    def smooth_data(self, data, window_length=13, polyorder=2):
        smoothed_df = data.copy()
        for col in ['deg1', 'deg2', 'deg3', 'deg4', 'degsec1', 'degsec2', 'degsec3', 'degsec4']:
            smoothed_df[col] = savgol_filter(data[col], window_length=window_length, polyorder=polyorder)
        return smoothed_df
    
    def normalize_time(self, target_trajectory, subject_trajectory):
        """궤적을 시간에 대해 정규화"""
        target_angles = target_trajectory[:, :4]
        subject_angles = subject_trajectory[:, :4]
        
        # 각도 데이터 정규화
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
        
    def model_based_interpolate_line(self, target, subject, interpolation_weight=0.5):
        """관절 관계를 고려한 모델 기반 선형 보간"""
        target_angles = target[:, :4]
        target_velocities = target[:, 4:]
        subject_angles = subject[:, :4]
        subject_velocities = subject[:, 4:]

        aligned_target, aligned_subject = self.normalize_time(target, subject)
        aligned_target_angles = aligned_target[:, :4]
        aligned_target_velocities = aligned_target[:, 4:]
        aligned_subject_angles = aligned_subject[:, :4]
        aligned_subject_velocities = aligned_subject[:, 4:]

        interpolated_degrees = np.zeros_like(aligned_target_angles)
        interpolated_velocities = np.zeros_like(aligned_target_velocities)

        for joint in range(4):
            for i in range(len(aligned_target_angles)):
                interpolated_degrees[i, joint] = (1 - interpolation_weight) * aligned_target_angles[i, joint] + interpolation_weight * aligned_subject_angles[i, joint]
                interpolated_velocities[i, joint] = (1 - interpolation_weight) * aligned_target_velocities[i, joint] + interpolation_weight * aligned_subject_velocities[i, joint]

        # 엔드이펙터 위치 계산
        endeffector_positions = np.array([calculate_end_effector_position(deg) for deg in interpolated_degrees])

        # 전체 데이터 결합
        combined_data = np.hstack([interpolated_degrees, interpolated_velocities, endeffector_positions])

        # Transformer 보정
        with torch.no_grad():
            segments = []
            segment_size = 100
            for i in range(0, len(interpolated_degrees), segment_size):
                segment = combined_data[i:i+segment_size]
                if len(segment) == 0:
                    continue
                # segment_tensor = torch.FloatTensor(segment).unsqueeze(0).to(self.device)
                # joint_interactions = self.model(segment_tensor).squeeze(0).cpu().numpy()
                segment_tensor = torch.FloatTensor(segment).unsqueeze(0).to(self.device)
                angles = segment_tensor[:, :, :4]  # 각도 추출
                velocities = segment_tensor[:, :, 4:8]  # 각속도 추출
                joint_interactions = self.model(angles, velocities).squeeze(0).cpu().numpy()
                segments.append(joint_interactions)
            
            if segments:
                model_output = np.vstack(segments)
                if len(model_output) > len(interpolated_degrees):
                    model_output = model_output[:len(interpolated_degrees)]
                
                correction_strength = 0.3
                for joint in range(4):
                    for i in range(len(interpolated_degrees)):
                        lower_bound = min(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                        upper_bound = max(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                        bounded_model_output = np.clip(model_output[i, joint], lower_bound, upper_bound)
                        original_val = interpolated_degrees[i, joint]
                        interpolated_degrees[i, joint] = (1 - correction_strength) * original_val + correction_strength * bounded_model_output
                        interpolated_degrees[i, joint] = np.clip(interpolated_degrees[i, joint], lower_bound, upper_bound)
        
        # 각속도 재계산
        for joint in range(4):
            interpolated_velocities[:, joint] = np.gradient(interpolated_degrees[:, joint])

        return interpolated_degrees, interpolated_velocities
    
    def model_based_interpolate_arc(self, target, subject):
        """관절 관계를 고려한 모델 기반 호 보간"""
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
        interpolation_weight = 0.5  # 기본 보간 가중치
        
        # 기본 Cubic Spline 보간 수행
        for joint in range(4):
            for i in range(n_points):
                # 각 시점의 선형 보간된 값 (경계 역할)
                lower_bound = min(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                upper_bound = max(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                
                # 3개의 제어점 사용
                control_indices = [0, n_points//2, -1]
                control_times = [0, 0.5, 1]
                
                # 제어점의 각도와 각속도 계산
                control_angles = np.zeros(len(control_indices))
                control_velocities = np.zeros(len(control_indices))
                
                for i, idx in enumerate(control_indices):
                    target_val = aligned_target_angles[idx, joint]
                    subject_val = aligned_subject_angles[idx, joint]
                    ctrl_lower = min(target_val, subject_val)
                    ctrl_upper = max(target_val, subject_val)
                    
                    # 제어점 계산 및 경계 내로 제한
                    control_angles[i] = np.clip(
                        (1 - interpolation_weight) * target_val + interpolation_weight * subject_val,
                        ctrl_lower, ctrl_upper
                    )
                    
                    target_vel = aligned_target_velocities[idx, joint]
                    subject_vel = aligned_subject_velocities[idx, joint]
                    control_velocities[i] = (1 - interpolation_weight) * target_vel + interpolation_weight * subject_vel
                
                # Cubic Spline 보간
                cs_angle = CubicSpline(control_times, control_angles)
                cs_vel = CubicSpline(control_times, control_velocities)
                
                # 보간된 값 계산 및 경계 내로 제한
                for i in range(n_points):
                    lower_bound = min(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                    upper_bound = max(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                    
                    # 스플라인 값 계산 및 경계 내로 클리핑
                    spline_val = cs_angle(t[i])
                    interpolated_degrees[i, joint] = np.clip(spline_val, lower_bound, upper_bound)
                    
                    vel_val = cs_vel(t[i])
                    interpolated_velocities[i, joint] = vel_val
        
        # 엔드이펙터 위치 계산
        endeffector_positions = np.array([calculate_end_effector_position(deg) for deg in interpolated_degrees])

        # 전체 데이터 결합
        combined_data = np.hstack([interpolated_degrees, interpolated_velocities, endeffector_positions])

        # Transformer 보정
        with torch.no_grad():
            segments = []
            segment_size = 100
            for i in range(0, n_points, segment_size):
                segment = combined_data[i:i+segment_size]
                if len(segment) == 0:
                    continue
                # segment_tensor = torch.FloatTensor(segment).unsqueeze(0).to(self.device)
                # joint_interactions = self.model(segment_tensor).squeeze(0).cpu().numpy()
                segment_tensor = torch.FloatTensor(segment).unsqueeze(0).to(self.device)
                angles = segment_tensor[:, :, :4]  # 각도 추출
                velocities = segment_tensor[:, :, 4:8]  # 각속도 추출
                joint_interactions = self.model(angles, velocities).squeeze(0).cpu().numpy()
                segments.append(joint_interactions)
            
            if segments:
                model_output = np.vstack(segments)
                if len(model_output) > len(interpolated_degrees):
                    model_output = model_output[:len(interpolated_degrees)]
                
                correction_strength = 0.25
                for joint in range(4):
                    for i in range(len(interpolated_degrees)):
                        lower_bound = min(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                        upper_bound = max(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                        bounded_model_output = np.clip(model_output[i, joint], lower_bound, upper_bound)
                        original_val = interpolated_degrees[i, joint]
                        interpolated_degrees[i, joint] = (1 - correction_strength) * original_val + correction_strength * bounded_model_output
                        interpolated_degrees[i, joint] = np.clip(interpolated_degrees[i, joint], lower_bound, upper_bound)
        
        # 각속도 재계산
        for joint in range(4):
            interpolated_velocities[:, joint] = np.gradient(interpolated_degrees[:, joint])

        return interpolated_degrees, interpolated_velocities

    def model_based_interpolate_circle(self, target, subject, correction_strength=0.2):
        aligned_target, aligned_subject = self.normalize_time(target, subject)
        aligned_target_angles = aligned_target[:, :4]
        aligned_subject_angles = aligned_subject[:, :4]
        n_points = len(aligned_target_angles)
        interpolated_degrees = np.zeros_like(aligned_target_angles)

        t = np.linspace(0, 1, n_points)
        weights = t * t * (3 - 2 * t)

        from scipy.spatial.transform import Rotation as R
        for i in range(n_points):
            target_rad = np.radians(aligned_target_angles[i, :3])
            subject_rad = np.radians(aligned_subject_angles[i, :3])
            q_target = R.from_euler('xyz', target_rad)
            q_subject = R.from_euler('xyz', subject_rad)
            q_target_arr = q_target.as_quat()
            q_subject_arr = q_subject.as_quat()

            # 방향성 보정
            dot = np.sum(q_target_arr * q_subject_arr)
            if dot < 0.0:
                q_subject_arr = -q_subject_arr
                dot = -dot

            if dot > 0.9995:
                result = q_target_arr + weights[i] * (q_subject_arr - q_target_arr)
            else:
                theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
                sin_theta_0 = np.sin(theta_0)
                if sin_theta_0 < 1e-6:  # 너무 작은 각도 방지
                    result = q_target_arr
                else:
                    theta = theta_0 * weights[i]
                    sin_theta = np.sin(theta)
                    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
                    s1 = sin_theta / sin_theta_0
                    result = s0 * q_target_arr + s1 * q_subject_arr
            result = result / np.linalg.norm(result)  # 정규화
            q_interp = R.from_quat(result)
            euler_angles = q_interp.as_euler('xyz', degrees=True)
            interpolated_degrees[i, :3] = euler_angles
            interpolated_degrees[i, 3] = (1 - weights[i]) * aligned_target_angles[i, 3] + weights[i] * aligned_subject_angles[i, 3]

        # 각속도 계산
        interpolated_velocities = np.gradient(interpolated_degrees, axis=0)

        # 엔드이펙터 위치 계산
        endeffector_positions = np.array([calculate_end_effector_position(deg) for deg in interpolated_degrees])

        # 전체 데이터 결합 (각도 4 + 각속도 4 + 위치 3 = 11)
        combined_data = np.hstack([interpolated_degrees, interpolated_velocities, endeffector_positions])

        # Transformer 보정
        with torch.no_grad():
            segment_tensor = torch.FloatTensor(combined_data).unsqueeze(0).to(self.device)
            angles = segment_tensor[:, :, :4]  # 각도 추출
            velocities = segment_tensor[:, :, 4:8]  # 각속도 추출
            joint_interactions = self.model(angles, velocities).squeeze(0).cpu().numpy()
            for joint in range(4):
                interpolated_degrees[:, joint] = (1 - correction_strength) * interpolated_degrees[:, joint] + correction_strength * joint_interactions[:, joint]

        interpolated_velocities = np.gradient(interpolated_degrees, axis=0)

        return interpolated_degrees, interpolated_velocities
        
    def interpolate_trajectory(self, target_df, user_df, trajectory_type):
        target_with_vel = target_df.copy()
        user_with_vel = user_df.copy()
        
        for df in [target_with_vel, user_with_vel]:
            df['degsec1'] = np.gradient(df['deg1'])
            df['degsec2'] = np.gradient(df['deg2'])
            df['degsec3'] = np.gradient(df['deg3'])
            df['degsec4'] = np.gradient(df['deg4'])
        
        target_smoothed = self.smooth_data(target_with_vel)
        user_smoothed = self.smooth_data(user_with_vel)

        target_angles = target_smoothed[['deg1', 'deg2', 'deg3', 'deg4']].values
        target_velocities = target_smoothed[['degsec1', 'degsec2', 'degsec3', 'degsec4']].values
        user_angles = user_smoothed[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_velocities = user_smoothed[['degsec1', 'degsec2', 'degsec3', 'degsec4']].values
        
        target_data = np.column_stack([target_angles, target_velocities])
        user_data = np.column_stack([user_angles, user_velocities])
        
        aligned_target, aligned_subject = self.normalize_time(target_data, user_data)
        aligned_target_angles = aligned_target[:, :4]
        aligned_target_velocities = aligned_target[:, 4:]
        aligned_subject_angles = aligned_subject[:, :4]
        aligned_subject_velocities = aligned_subject[:, 4:]
        
        n_points = len(aligned_target_angles)
        assert len(aligned_target_angles) == len(aligned_subject_angles), "Aligned lengths do not match after DTW"

        timestamps = np.arange(n_points).reshape(-1, 1) / n_points
        
        t = np.linspace(0, 1, n_points)
        target_bias = 0.7
        weights = target_bias + (1 - target_bias) * (t * t * (3 - 2 * t))
        
        interpolated_angles = (1 - weights[:, np.newaxis]) * aligned_target_angles + weights[:, np.newaxis] * aligned_subject_angles
        interpolated_velocities = (1 - weights[:, np.newaxis]) * aligned_target_velocities + weights[:, np.newaxis] * aligned_subject_velocities
        
        interpolated_positions = np.array([calculate_end_effector_position(deg) for deg in interpolated_angles])
        
        # 입력 데이터 준비: 각도 4 + 각속도 4 + 위치 3 + timestamp 1 = 12
        combined_input = np.hstack([interpolated_angles, interpolated_velocities, interpolated_positions[:, :3], timestamps])
        with torch.no_grad():
            input_tensor = torch.FloatTensor(combined_input).unsqueeze(0).to(self.device)  # (1, n_points, 12)
            angles = input_tensor[:, :, :4]
            velocities = input_tensor[:, :, 4:8]
            timestamps_tensor = input_tensor[:, :, 11:12]  # timestamp 위치 조정
            output_angles = self.model(angles, velocities, timestamps_tensor).squeeze(0).cpu().numpy()
        
        interpolated_degrees = output_angles
        interpolated_velocities = np.gradient(interpolated_degrees, axis=0)
        
        result_df = pd.DataFrame(interpolated_degrees, columns=['deg1', 'deg2', 'deg3', 'deg4'])
        result_df['degsec1'] = interpolated_velocities[:, 0]
        result_df['degsec2'] = interpolated_velocities[:, 1]
        result_df['degsec3'] = interpolated_velocities[:, 2]
        result_df['degsec4'] = interpolated_velocities[:, 3]
        result_df[['x_end', 'y_end', 'z_end']] = interpolated_positions * 1000
        
        return result_df

    def visualize_trajectories(self, target_df, user_df, generated_df, trajectory_type, generation_number=1):
        """타겟과 사용자 궤적과 생성된 궤적을 시각화"""
        # 관절 각도 데이터 준비
        target_degrees = target_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_degrees = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        generated_degrees = generated_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        
        # End-effector 위치 계산 - 시각화를 위한 변환 적용
        target_degrees_adj = target_degrees.copy()
        target_degrees_adj[:, 1] -= 90
        target_degrees_adj[:, 3] -= 90
        
        user_degrees_adj = user_degrees.copy()
        user_degrees_adj[:, 1] -= 90
        user_degrees_adj[:, 3] -= 90
        
        target_ends = np.array([calculate_end_effector_position(deg) for deg in target_degrees_adj]) * 1000
        user_ends = np.array([calculate_end_effector_position(deg) for deg in user_degrees_adj]) * 1000

        # 시각화
        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(2, 3, figure=fig)
        
        # 3D end-effector 궤적 시각화
        ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
        ax_3d.plot(target_ends[:, 0], target_ends[:, 1], target_ends[:, 2], 
                'b-', label='target trajectory')
        ax_3d.plot(user_ends[:, 0], user_ends[:, 1], user_ends[:, 2],
                'r-', label='user trajectory')
        ax_3d.plot(generated_df['x_end'], generated_df['y_end'], generated_df['z_end'], 
                'g-', label='interpolated trajectory')
        
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title('End-effector Trajectory')
        ax_3d.legend()

        # 관절별 그래프
        target_time = np.arange(len(target_degrees))
        user_time = np.arange(len(user_degrees))
        aligned_time = np.arange(len(generated_degrees))
        joint_titles = ['deg1', 'deg2', 'deg3', 'deg4']
        for idx, joint in enumerate(['deg1', 'deg2', 'deg3', 'deg4']):
            row = idx // 2
            col = (idx % 2) + 1
            
            ax = fig.add_subplot(gs[row, col])
            
            ax.plot(target_time, target_df[joint], 'b--', label='target')
            ax.plot(user_time, user_df[joint], 'r:', label='user')
            ax.plot(aligned_time, generated_df[joint], 'g-', label='interpolate')
            
            ax.set_title(joint_titles[idx])
            ax.set_xlabel('timestamp')
            ax.set_ylabel('deg')
            ax.grid(True)
            ax.legend()

            # 영역 표시 - 두 궤적 사이의 허용 범위
            for i in range(min(len(target_time), len(user_time))):
                if i < len(target_time) and i < len(user_time):
                    lower = min(target_df[joint].iloc[i], user_df[joint].iloc[i])
                    upper = max(target_df[joint].iloc[i], user_df[joint].iloc[i])
                    ax.fill_between([i, i+1], [lower, lower], [upper, upper], color='gray', alpha=0.2)
            
            # 경계를 벗어난 지점 확인 및 표시
            for i in range(len(generated_df)):
                if i < len(target_df) and i < len(user_df):
                    lower = min(target_df[joint].iloc[i], user_df[joint].iloc[i])
                    upper = max(target_df[joint].iloc[i], user_df[joint].iloc[i])
                    val = generated_df[joint].iloc[i]
                    
                    # 범위를 벗어났는지 확인
                    if val < lower or val > upper:
                        ax.plot(i, val, 'rx', markersize=8)  # 경계를 벗어난 점 표시

        fig.suptitle(f'Trajectory_type: {trajectory_type}', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
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
        full_df['timestamp'] = [i * 10 for i in range(num_points)]  # 10ms 간격
        
        # 각도, 엔드이펙터의 데이터 설정
        full_df['deg'] = (generated_df['deg1'].round(3).astype(str) + '/' + 
                        generated_df['deg2'].round(3).astype(str) + '/' +
                        generated_df['deg3'].round(3).astype(str) + '/' +
                        generated_df['deg4'].round(3).astype(str))

        full_df['endpoint'] = (generated_df['x_end'].round(3).astype(str) + '/' + 
                            generated_df['y_end'].round(3).astype(str) + '/' + 
                            generated_df['z_end'].round(3).astype(str))
        
        # 실제 파일 형식에 맞게 열 순서 및 포맷 조정
        full_df = full_df[['r', 'sequence', 'timestamp', 'deg', 'endpoint']]
        
        full_df.to_csv(generation_path, index=False)
        print(f"\nCompleted saving the generated trajectory: {generation_path}")

        return generation_path
        
    # def analyze_joint_relationships(self):
    #     """관절 간 상관관계 분석 및 시각화"""
    #     # 모델의 가중치 분석
    #     # 첫 번째 joint attention layer의 가중치 추출
    #     with torch.no_grad():
    #         # 관절 간 관계 가시화를 위한 간단한 입력 데이터 생성
    #         dummy_input = torch.eye(4).unsqueeze(0).to(self.device)
    #         dummy_input = self.model.joint_embedding(dummy_input)
            
    #         # Joint Attention의 query, key, value 가중치 추출
    #         attentions = []
    #         for layer in self.model.joint_attention_layers:
    #             Q = layer.query(dummy_input)
    #             K = layer.key(dummy_input)
    #             scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(layer.d_model).float())
    #             attention = torch.softmax(scores, dim=-1)
    #             attentions.append(attention.squeeze(0).cpu().numpy())
        
    #     # 관절 간 관계 시각화
    #     fig, axes = plt.subplots(1, len(attentions), figsize=(len(attentions) * 5, 5))
    #     joint_names = ['deg1', 'deg2', 'deg3', 'deg4']
        
    #     if len(attentions) == 1:
    #         axes = [axes]
            
    #     for i, attn in enumerate(attentions):
    #         ax = axes[i]
    #         im = ax.imshow(attn, cmap='viridis')
    #         ax.set_title(f'Joint Attention Layer {i+1}')
    #         ax.set_xticks(range(4))
    #         ax.set_yticks(range(4))
    #         ax.set_xticklabels(joint_names)
    #         ax.set_yticklabels(joint_names)
            
    #         # 각 셀에 값 표시
    #         for j in range(4):
    #             for k in range(4):
    #                 text = ax.text(k, j, f'{attn[j, k]:.2f}',
    #                               ha="center", va="center", color="w" if attn[j, k] > 0.5 else "black")
            
    #         fig.colorbar(im, ax=ax)
        
    #     plt.tight_layout()
    #     plt.suptitle('Interarticular Correlation Analysis', fontsize=16)
    #     plt.subplots_adjust(top=0.85)
    #     plt.show()
        
    #     # 가장 강한 관계 찾기
    #     avg_attention = np.mean(np.array(attentions), axis=0)
    #     np.fill_diagonal(avg_attention, 0) 
        
    #     strongest_pairs = []
    #     for i in range(4):
    #         for j in range(i+1, 4):
    #             strongest_pairs.append((i, j, avg_attention[i, j]))
        
    #     # 상관관계가 강한 순서로 정렬
    #     strongest_pairs.sort(key=lambda x: x[2], reverse=True)
        
    #     print("\nStrongest Joint Correlation (Top 3):")
    #     for i, (joint1, joint2, strength) in enumerate(strongest_pairs[:3]):
    #         print(f"{i+1}. {joint_names[joint1]} ↔ {joint_names[joint2]}: {strength:.4f}")