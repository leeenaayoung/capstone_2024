import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from fastdtw import fastdtw
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from torch import nn
from analyzer import TrajectoryAnalyzer
from utils import calculate_end_effector_position
from generation_m_model import JointTrajectoryTransformer
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import UnivariateSpline
import scipy.stats as stats
from scipy.interpolate import BSpline, splev, splrep

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
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
        
        # 관절 제한 설정
        self.joint_limits = {   
            0: (-10, 110),
            1: (0, 150),     
            2: (0, 150),     
            3: (-90, 90)     
        }

        self.model.eval()
    
    # def smooth_data(self, data, R=0.02, Q=0.1):
    #     """칼만 필터를 사용한 데이터 스무딩"""
    #     angles = data[['deg1', 'deg2', 'deg3', 'deg4']].values
    #     velocities = data[['degsec1', 'degsec2', 'degsec3', 'degsec4']].values
    #     n_samples, n_joints = angles.shape
        
    #     smoothed_angles = np.zeros_like(angles)
    #     smoothed_velocities = np.zeros_like(velocities)
        
    #     for joint in range(n_joints):   
    #         x_hat_full = np.array([angles[0, joint], velocities[0, joint]])
    #         P_full = np.eye(2)
            
    #         dt = 1.0
    #         A = np.array([[1, dt],
    #                     [0, 1]])
    #         H = np.eye(2) 
            
    #         Q_matrix = Q * np.array([[dt**4/4, dt**3/2],
    #                                 [dt**3/2, dt**2]])
    #         R_matrix = np.diag([R, R*10]) 
            
    #         smoothed_angles[0, joint] = x_hat_full[0]
    #         smoothed_velocities[0, joint] = x_hat_full[1]
            
    #         for k in range(1, n_samples):
    #             x_hat_full = A @ x_hat_full
    #             P_full = A @ P_full @ A.T + Q_matrix
                
    #             z = np.array([angles[k, joint], velocities[k, joint]])
                
    #             y = z - H @ x_hat_full
    #             S = H @ P_full @ H.T + R_matrix
    #             K = P_full @ H.T @ np.linalg.inv(S)
                
    #             x_hat_full = x_hat_full + K @ y
    #             P_full = (np.eye(2) - K @ H) @ P_full
                
    #             smoothed_angles[k, joint] = x_hat_full[0]
    #             smoothed_velocities[k, joint] = x_hat_full[1]
        
    #     smoothed_df = data.copy()
    #     smoothed_df[['deg1', 'deg2', 'deg3', 'deg4']] = smoothed_angles
    #     smoothed_df[['degsec1', 'degsec2', 'degsec3', 'degsec4']] = smoothed_velocities
        
    #     return smoothed_df

    def smooth_data(self, data, smoothing_factor=0.5, degree=3):
        """
        스플라인 스무딩을 사용하여 로봇 관절 각도와 각속도 데이터를 스무딩합니다.
        """
        # 원본 데이터 복사
        smoothed_df = data.copy()
        
        # 각 열에 대해 스플라인 스무딩 적용
        for col in ['deg1', 'deg2', 'deg3', 'deg4', 'degsec1', 'degsec2', 'degsec3', 'degsec4']:
            if col in data.columns:
                # 시간 인덱스 생성 (데이터 포인트의 순서대로)
                x = np.arange(len(data))
                y = data[col].values
                
                # 데이터의 규모에 따라 스무딩 매개변수 조정
                # 데이터 포인트가 많을수록 s 값을 키워야 효과적인 스무딩 발생
                s = smoothing_factor * len(data)
                
                # UnivariateSpline으로 스무딩 수행
                # s 매개변수는 스플라인과 데이터 사이의 제곱 편차 합에 대한 상한을 제어
                spline = UnivariateSpline(x, y, k=degree, s=s)
                
                # 스무딩된 값으로 데이터프레임 갱신
                smoothed_df[col] = spline(x)
                
                # 각속도 열을 스무딩할 때 물리적 제약 고려
                # 각속도는 각도의 미분이므로 일관성을 유지하는 것이 중요
                if col.startswith('degsec'):
                    # 해당 각도 열 찾기 (예: degsec1 → deg1)
                    angle_col = 'deg' + col[6:]
                    if angle_col in data.columns:
                        # 각도 데이터의 수치 미분이 스무딩된 각속도와 일치하도록 조정 가능
                        # 이 주석을 해제하면 각도와 각속도 간의 물리적 일관성을 더 강제할 수 있음
                        # smoothed_df[col] = np.gradient(smoothed_df[angle_col].values)
                        pass
        
        return smoothed_df
    
    # dtw 사용 정규화(기존)
    # def normalize_time(self, target_trajectory, subject_trajectory):
    #     """선형 궤적 시간 정규화"""
    #     target_angles = target_trajectory[:, :4]
    #     subject_angles = subject_trajectory[:, :4]
        
    #     normalized_target_angles = np.zeros_like(target_angles)
    #     normalized_subject_angles = np.zeros_like(subject_angles)
        
    #     for joint in range(4):
    #         min_val = self.joint_limits[joint][0] 
    #         max_val = self.joint_limits[joint][1] 
    #         range_val = max_val - min_val
    #         normalized_target_angles[:, joint] = (target_angles[:, joint] - min_val) / range_val
    #         normalized_subject_angles[:, joint] = (subject_angles[:, joint] - min_val) / range_val
        
    #     _, path = fastdtw(normalized_target_angles, normalized_subject_angles, dist=euclidean)
    #     path = np.array(path, dtype=np.int32)
        
    #     aligned_target = target_trajectory[path[:, 0]]
    #     aligned_subject = subject_trajectory[path[:, 1]]
        
    #     return aligned_target, aligned_subject
            
    def normalize_time(self, target_trajectory, subject_trajectory):
        """궤적 유형에 따라 적절한 정규화 방법 호출"""
        if hasattr(self, 'current_type'):
            if 'clock' in self.current_type.lower() or 'counter' in self.current_type.lower():
                return self.normalize_time_circle(target_trajectory, subject_trajectory)
            elif 'v_' in self.current_type.lower() or 'h_' in self.current_type.lower():
                return self.normalize_time_arc(target_trajectory, subject_trajectory)
        
        # 기본값은 DTW
        return self.normalize_time_dtw(target_trajectory, subject_trajectory)
    
    def normalize_time_dtw(self, target_trajectory, subject_trajectory):
        """선형 궤적 시간 정규화"""
        target_angles = target_trajectory[:, :4]
        subject_angles = subject_trajectory[:, :4]
        
        normalized_target_angles = np.zeros_like(target_angles)
        normalized_subject_angles = np.zeros_like(subject_angles)
        
        for joint in range(4):
            min_val = self.joint_limits[joint][0] 
            max_val = self.joint_limits[joint][1] 
            range_val = max_val - min_val
            normalized_target_angles[:, joint] = (target_angles[:, joint] - min_val) / range_val
            normalized_subject_angles[:, joint] = (subject_angles[:, joint] - min_val) / range_val
        
        _, path = fastdtw(normalized_target_angles, normalized_subject_angles, dist=euclidean)
        path = np.array(path, dtype=np.int32)
        
        aligned_target = target_trajectory[path[:, 0]]
        aligned_subject = subject_trajectory[path[:, 1]]
        
        return aligned_target, aligned_subject
    
    def normalize_time_arc(self, target_trajectory, subject_trajectory, num_points=None):
        """ 호 길이 기반 시간 정규화 (호 궤적 전용) """
        # if target_trajectory.ndim != 2 or subject_trajectory.ndim != 2:
        #     raise ValueError(f"입력 데이터는 2차원 배열이어야 합니다. 현재 차원: target {target_trajectory.ndim}, subject {subject_trajectory.ndim}")
        
        # if target_trajectory.shape[1] != subject_trajectory.shape[1]:
        #     raise ValueError(f"두 궤적의 특성 수가 일치해야 합니다. target: {target_trajectory.shape[1]}, subject: {subject_trajectory.shape[1]}")
        
        # 기본 샘플 수 설정 (더 긴 궤적에 맞춤)
        if num_points is None:
            num_points = max(len(target_trajectory), len(subject_trajectory))
        
        # 각도와 각속도 분리
        target_angles = target_trajectory[:, :4]
        target_velocities = target_trajectory[:, 4:]
        subject_angles = subject_trajectory[:, :4]
        subject_velocities = subject_trajectory[:, 4:]
        
        # 1. 엔드이펙터 위치 계산
        target_ee_positions = np.zeros((len(target_angles), 3))
        subject_ee_positions = np.zeros((len(subject_angles), 3))
        
        for i in range(len(target_angles)):
            angles_adj = target_angles[i].copy()
            # angles_adj[1] -= 90
            # angles_adj[3] -= 90
            target_ee_positions[i] = calculate_end_effector_position(angles_adj)
        
        for i in range(len(subject_angles)):
            angles_adj = subject_angles[i].copy()
            # angles_adj[1] -= 90
            # angles_adj[3] -= 90
            subject_ee_positions[i] = calculate_end_effector_position(angles_adj)
        
        # 2. 호 길이 기반 매개변수화
        # 각 점 사이의 거리를 누적하여 호 길이 계산
        def arc_length_parameterization(positions):
            arc_lengths = np.zeros(len(positions))
            for i in range(1, len(positions)):
                # 현재 점과 이전 점 사이의 유클리드 거리 계산
                arc_lengths[i] = arc_lengths[i-1] + np.linalg.norm(positions[i] - positions[i-1])
            
            # 정규화된 매개변수 (0에서 1)
            if arc_lengths[-1] > 0:
                params = arc_lengths / arc_lengths[-1]
            else:
                params = np.linspace(0, 1, len(positions))
            
            return params
        
        # 타겟과 사용자 궤적의 호 길이 매개변수화
        target_params = arc_length_parameterization(target_ee_positions)
        subject_params = arc_length_parameterization(subject_ee_positions)
        
        # 3. 균일한 매개변수 간격으로 샘플링할 포인트
        uniform_params = np.linspace(0, 1, num_points)
        
        # 재샘플링된 궤적을 저장할 배열
        resampled_target = np.zeros((num_points, target_trajectory.shape[1]))
        resampled_subject = np.zeros((num_points, subject_trajectory.shape[1]))
        
        # 4. 각 차원(관절 각도, 각속도)별로 스플라인 피팅 및 리샘플링
        # 호 궤적에는 CubicSpline이 더 적합할 수 있음 (엔드포인트 연속성)
        from scipy.interpolate import CubicSpline
        
        for dim in range(target_trajectory.shape[1]):
            # 타겟 궤적 스플라인 피팅
            target_spline = CubicSpline(
                target_params, 
                target_trajectory[:, dim],
                bc_type='natural'  # 자연 경계 조건 (2차 미분 = 0)
            )
            resampled_target[:, dim] = target_spline(uniform_params)
            
            # 사용자 궤적 스플라인 피팅
            subject_spline = CubicSpline(
                subject_params, 
                subject_trajectory[:, dim],
                bc_type='natural'
            )
            resampled_subject[:, dim] = subject_spline(uniform_params)
        
        # 5. 물리적 관절 제한만 적용 (클리핑 완화)
        for joint in range(4):
            # 물리적 제한
            joint_min = self.joint_limits[joint][0]
            joint_max = self.joint_limits[joint][1]
            
            # 물리적 한계 내로만 클리핑
            resampled_target[:, joint] = np.clip(resampled_target[:, joint], joint_min, joint_max)
            resampled_subject[:, joint] = np.clip(resampled_subject[:, joint], joint_min, joint_max)
        
        return resampled_target, resampled_subject


    def normalize_time_circle(self, target_trajectory, subject_trajectory, num_points=None):
        """
        위상 기반 시간 정규화 (원형 궤적 전용)
        
        Parameters:
        target_trajectory: 타겟 궤적 데이터 (각도와 각속도)
        subject_trajectory: 사용자 궤적 데이터 (각도와 각속도)
        num_points: 정규화 후 데이터 포인트 수
        
        Returns:
        tuple: (정규화된 타겟 궤적, 정규화된 사용자 궤적)
        """
        if target_trajectory.ndim != 2 or subject_trajectory.ndim != 2:
            raise ValueError(f"입력 데이터는 2차원 배열이어야 합니다. 현재 차원: target {target_trajectory.ndim}, subject {subject_trajectory.ndim}")
        
        if target_trajectory.shape[1] != subject_trajectory.shape[1]:
            raise ValueError(f"두 궤적의 특성 수가 일치해야 합니다. target: {target_trajectory.shape[1]}, subject: {subject_trajectory.shape[1]}")
        
        # 기본 샘플 수 설정 (더 긴 궤적에 맞춤)
        if num_points is None:
            num_points = max(len(target_trajectory), len(subject_trajectory))
        
        # 각도와 각속도 분리
        target_angles = target_trajectory[:, :4]
        target_velocities = target_trajectory[:, 4:]
        subject_angles = subject_trajectory[:, :4]
        subject_velocities = subject_trajectory[:, 4:]
        
        # 1. 엔드이펙터 위치 계산
        target_ee_positions = np.zeros((len(target_angles), 3))
        subject_ee_positions = np.zeros((len(subject_angles), 3))
        
        for i in range(len(target_angles)):
            angles_adj = target_angles[i].copy()
            # angles_adj[1] -= 90
            # angles_adj[3] -= 90
            target_ee_positions[i] = calculate_end_effector_position(angles_adj)
        
        for i in range(len(subject_angles)):
            angles_adj = subject_angles[i].copy()
            # angles_adj[1] -= 90
            # angles_adj[3] -= 90
            subject_ee_positions[i] = calculate_end_effector_position(angles_adj)
        
        # 2. 원형 궤적 매개변수화 (위상 기반)
        # 중심점 추정
        def estimate_circle_center(points):
            # 원형 궤적의 중심은 XY 평면상 점들의 평균으로 근사
            x_mean = np.mean(points[:, 0])
            y_mean = np.mean(points[:, 1])
            return np.array([x_mean, y_mean])
        
        target_center = estimate_circle_center(target_ee_positions[:, :2])
        subject_center = estimate_circle_center(subject_ee_positions[:, :2])
        
        # 위상각 계산
        def compute_phase_angles(points, center):
            # 중심점을 기준으로 각 점의 위상각 계산
            centered_points = points[:, :2] - center
            # arctan2는 -π에서 π 사이의 각도 반환
            phase_angles = np.arctan2(centered_points[:, 1], centered_points[:, 0])
            # unwrap은 각도 불연속점을 연속적으로 만듦 (-π/π 경계에서 2π 추가)
            return np.unwrap(phase_angles)
        
        target_phases = compute_phase_angles(target_ee_positions, target_center)
        subject_phases = compute_phase_angles(subject_ee_positions, subject_center)
        
        # 위상각 정규화 (0부터 1로)
        # 시작점을 0으로, 끝점을 1로 정규화
        if len(target_phases) > 1:
            target_params = (target_phases - target_phases.min()) / (target_phases.max() - target_phases.min())
        else:
            target_params = np.array([0])
            
        if len(subject_phases) > 1:
            subject_params = (subject_phases - subject_phases.min()) / (subject_phases.max() - subject_phases.min())
        else:
            subject_params = np.array([0])
        
        # 3. 균일한 매개변수 간격으로 샘플링할 포인트
        uniform_params = np.linspace(0, 1, num_points)
        
        # 재샘플링된 궤적을 저장할 배열
        resampled_target = np.zeros((num_points, target_trajectory.shape[1]))
        resampled_subject = np.zeros((num_points, subject_trajectory.shape[1]))
        
        # 4. 각 차원(관절 각도, 각속도)별로 스플라인 피팅 및 리샘플링
        # 원형 궤적은 주기성 고려 필요
        for dim in range(target_trajectory.shape[1]):
            # 첫번째와 마지막 값이 비슷한지 확인 (주기성 체크)
            target_is_periodic = False
            if len(target_trajectory) > 2:
                target_first_last_diff = abs(target_trajectory[0, dim] - target_trajectory[-1, dim])
                target_range = max(target_trajectory[:, dim]) - min(target_trajectory[:, dim])
                if target_range > 0 and target_first_last_diff / target_range < 0.1:  # 10% 이내면 주기적
                    target_is_periodic = True
            
            subject_is_periodic = False
            if len(subject_trajectory) > 2:
                subject_first_last_diff = abs(subject_trajectory[0, dim] - subject_trajectory[-1, dim])
                subject_range = max(subject_trajectory[:, dim]) - min(subject_trajectory[:, dim])
                if subject_range > 0 and subject_first_last_diff / subject_range < 0.1:
                    subject_is_periodic = True
            
            # 타겟 궤적 스플라인 피팅
            if target_is_periodic:
                # 주기적인 경우 확장된 데이터로 피팅
                extended_params = np.concatenate([target_params, [target_params[-1] + (target_params[1] - target_params[0])]])
                extended_values = np.concatenate([target_trajectory[:, dim], [target_trajectory[0, dim]]])
                
                # 주기성 고려한 스플라인
                from scipy.interpolate import CubicSpline
                target_spline = CubicSpline(
                    extended_params, 
                    extended_values,
                    bc_type='periodic'  # 주기적 경계 조건
                )
            else:
                # 비주기적인 경우 일반 스플라인
                from scipy.interpolate import UnivariateSpline
                target_spline = UnivariateSpline(
                    target_params, 
                    target_trajectory[:, dim], 
                    k=3,  # 3차 스플라인
                    s=0.1 * len(target_params)  # 약간의 스무딩 적용
                )
            
            # 사용자 궤적 스플라인 피팅
            if subject_is_periodic:
                # 주기적인 경우 확장된 데이터로 피팅
                extended_params = np.concatenate([subject_params, [subject_params[-1] + (subject_params[1] - subject_params[0])]])
                extended_values = np.concatenate([subject_trajectory[:, dim], [subject_trajectory[0, dim]]])
                
                # 주기성 고려한 스플라인
                from scipy.interpolate import CubicSpline
                subject_spline = CubicSpline(
                    extended_params, 
                    extended_values,
                    bc_type='periodic'  # 주기적 경계 조건
                )
            else:
                # 비주기적인 경우 일반 스플라인
                from scipy.interpolate import UnivariateSpline
                subject_spline = UnivariateSpline(
                    subject_params, 
                    subject_trajectory[:, dim], 
                    k=3,
                    s=0.1 * len(subject_params)
                )
            
            # 균일 간격으로 리샘플링
            resampled_target[:, dim] = target_spline(uniform_params)
            resampled_subject[:, dim] = subject_spline(uniform_params)
        
        # 5. 물리적 관절 제한만 적용 (클리핑 완화)
        for joint in range(4):
            # 물리적 제한
            joint_min = self.joint_limits[joint][0]
            joint_max = self.joint_limits[joint][1]
            
            # 물리적 한계 내로만 클리핑
            resampled_target[:, joint] = np.clip(resampled_target[:, joint], joint_min, joint_max)
            resampled_subject[:, joint] = np.clip(resampled_subject[:, joint], joint_min, joint_max)
        
        return resampled_target, resampled_subject


    def model_based_interpolate_line(self, target, subject, interpolation_weight=0.5):
        """관절 관계를 고려한 모델 기반 선형 보간"""
        aligned_target, aligned_subject = self.normalize_time(target, subject)
        
        aligned_target_angles = aligned_target[:, :4]
        aligned_target_velocities = aligned_target[:, 4:]
        aligned_subject_angles = aligned_subject[:, :4]
        aligned_subject_velocities = aligned_subject[:, 4:]

        n_points = len(aligned_target_angles)
        interpolated_degrees = np.zeros_like(aligned_target_angles)
        interpolated_velocities = np.zeros_like(aligned_target_velocities)

        for joint in range(4):
            for i in range(n_points):
                interpolated_degrees[i, joint] = (1 - interpolation_weight) * aligned_target_angles[i, joint] + \
                                            interpolation_weight * aligned_subject_angles[i, joint]
                interpolated_velocities[i, joint] = (1 - interpolation_weight) * aligned_target_velocities[i, joint] + \
                                                interpolation_weight * aligned_subject_velocities[i, joint]

        # 모델 입력 데이터를 각도 + 각속도로 결합
        model_input = np.column_stack([interpolated_degrees, interpolated_velocities])  # (n_points, 8)

        # 모델 기반 관절 간 상호작용 적용
        with torch.no_grad():
            segments = []
            segment_size = 100
            
            for i in range(0, n_points, segment_size):
                segment = model_input[i:i+segment_size]  # (segment_size, 8)
                if len(segment) == 0:
                    continue
                segment_tensor = torch.FloatTensor(segment).unsqueeze(0).to(self.device)  # (1, segment_size, 8)
                joint_interactions = self.model(segment_tensor).squeeze(0).cpu().numpy()  # (segment_size, 4)
                segments.append(joint_interactions)
            
            if segments:
                model_output = np.vstack(segments)  # (n_points, 4)
                if len(model_output) > len(interpolated_degrees):
                    model_output = model_output[:len(interpolated_degrees)]
                
                correction_strength = 0.3
                for joint in range(4):
                    for i in range(len(interpolated_degrees)):
                        # lower_bound = min(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                        # upper_bound = max(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                        # bounded_model_output = np.clip(model_output[i, joint], lower_bound, upper_bound)
                        # original_val = interpolated_degrees[i, joint]
                        # interpolated_degrees[i, joint] = (1 - correction_strength) * original_val + correction_strength * bounded_model_output
                        # interpolated_degrees[i, joint] = np.clip(interpolated_degrees[i, joint], lower_bound, upper_bound)
                        original_val = interpolated_degrees[i, joint]
                        interpolated_degrees[i, joint] = (1 - correction_strength) * original_val + correction_strength * model_output[i, joint]

        # 각속도 재계산
        for joint in range(4):
            interpolated_velocities[:, joint] = np.gradient(interpolated_degrees[:, joint])

        return interpolated_degrees, interpolated_velocities

    def model_based_interpolate_arc(self, target, subject, interpolation_weight=0.5):
        """관절 관계를 고려한 모델 기반 호 보간 - 확장된 허용 범위 클리핑"""
        aligned_target, aligned_subject = self.normalize_time(target, subject)
        
        aligned_target_angles = aligned_target[:, :4]
        aligned_target_velocities = aligned_target[:, 4:]
        aligned_subject_angles = aligned_subject[:, :4]
        aligned_subject_velocities = aligned_subject[:, 4:]

        n_points = len(aligned_target_angles)
        interpolated_degrees = np.zeros_like(aligned_target_angles)
        interpolated_velocities = np.zeros_like(aligned_target_velocities)
        
        # 균일한 시간 매개변수
        t = np.linspace(0, 1, n_points)
        
        # 부드러운 가중치 함수 (사인 기반 이징)
        ease_weight = 0.5 - 0.5 * np.cos(interpolation_weight * np.pi)
        
        # 각 관절별 스플라인 보간
        for joint in range(4):
            # 더 많은 제어점 사용 (부드러운 곡선을 위해)
            control_indices = [0, n_points//4, n_points//2, 3*n_points//4, n_points-1]
            control_times = [0, 0.25, 0.5, 0.75, 1]
            control_angles = []
            control_velocities = []
            
            # 제어점 값 계산
            for ci in control_indices:
                target_val = aligned_target_angles[ci, joint]
                subject_val = aligned_subject_angles[ci, joint]
                
                # 가중치 보간 (클리핑 없음)
                control_angles.append((1 - ease_weight) * target_val + ease_weight * subject_val)
                
                # 속도 제어점
                target_vel = aligned_target_velocities[ci, joint]
                subject_vel = aligned_subject_velocities[ci, joint]
                control_velocities.append((1 - ease_weight) * target_vel + ease_weight * subject_vel)
            
            # 자연 스플라인 보간
            cs_angle = CubicSpline(control_times, control_angles, bc_type='natural')
            cs_vel = CubicSpline(control_times, control_velocities, bc_type='natural')
            
            # 균일한 시간 간격으로 보간
            interpolated_degrees[:, joint] = cs_angle(t)
            interpolated_velocities[:, joint] = cs_vel(t)
        
        # 모델 입력 데이터를 각도 + 각속도로 결합
        model_input = np.column_stack([interpolated_degrees, interpolated_velocities])

        # 모델 기반 관절 간 상호작용 적용
        with torch.no_grad():
            segments = []
            segment_size = 100
            
            for i in range(0, n_points, segment_size):
                segment = model_input[i:i+segment_size]
                if len(segment) == 0:
                    continue
                segment_tensor = torch.FloatTensor(segment).unsqueeze(0).to(self.device)
                joint_interactions = self.model(segment_tensor).squeeze(0).cpu().numpy()
                segments.append(joint_interactions)
            
            if segments:
                model_output = np.vstack(segments)
                if len(model_output) > len(interpolated_degrees):
                    model_output = model_output[:len(interpolated_degrees)]
                
                # 모델 영향력 적용 (클리핑 없음)
                correction_strength = 0.2
                for joint in range(4):
                    interpolated_degrees[:, joint] = (1 - correction_strength) * interpolated_degrees[:, joint] + correction_strength * model_output[:, joint]
        
        # 확장된 허용 범위 클리핑 적용
        for joint in range(4):
            for i in range(len(interpolated_degrees)):
                if i < len(aligned_target_angles) and i < len(aligned_subject_angles):
                    # 타겟과 사용자 궤적 사이의 범위 계산
                    lower_bound = min(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                    upper_bound = max(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                    
                    # 범위 확장 (30% 여유)
                    range_width = upper_bound - lower_bound
                    if range_width > 0:
                        extended_lower = lower_bound - range_width * 0.3
                        extended_upper = upper_bound + range_width * 0.3
                    else:
                        # 두 값이 같은 경우 기본 여유 제공
                        extended_lower = lower_bound - 2
                        extended_upper = upper_bound + 2
                    
                    # 확장된 범위와 물리적 관절 한계 중 더 제한적인 범위 선택
                    extended_lower = max(extended_lower, self.joint_limits[joint][0])
                    extended_upper = min(extended_upper, self.joint_limits[joint][1])
                    
                    # 확장된 범위를 벗어나면 클리핑
                    if interpolated_degrees[i, joint] < extended_lower or interpolated_degrees[i, joint] > extended_upper:
                        interpolated_degrees[i, joint] = np.clip(interpolated_degrees[i, joint], extended_lower, extended_upper)
        
        # 강화된 글로벌 스무딩 적용
        # from scipy.signal import savgol_filter
        # window_length = min(101, n_points // 2 * 2 + 1)  # 훨씬 더 큰 윈도우 사용
        # window_length = max(5, window_length)  # 최소 5
        
        # for joint in range(4):
        #     # 매우 강한 스무딩 적용
        #     interpolated_degrees[:, joint] = savgol_filter(
        #         interpolated_degrees[:, joint], 
        #         window_length, 
        #         3,  # 3차 다항식
        #         mode='interp'  # 끝부분 외삽법 처리
        #     )
        
        # 각속도 재계산
        for joint in range(4):
            interpolated_velocities[:, joint] = np.gradient(interpolated_degrees[:, joint])

        return interpolated_degrees, interpolated_velocities


    def model_based_interpolate_circle(self, target, subject, interpolation_weight=0.5):
        """관절 관계를 고려한 모델 기반 원형 보간 - 확장된 허용 범위 클리핑"""
        from scipy.spatial.transform import Rotation as R
        
        aligned_target, aligned_subject = self.normalize_time(target, subject)
        
        aligned_target_angles = aligned_target[:, :4]
        aligned_target_velocities = aligned_target[:, 4:]
        aligned_subject_angles = aligned_subject[:, :4]
        aligned_subject_velocities = aligned_subject[:, 4:]

        n_points = len(aligned_target_angles)
        interpolated_degrees = np.zeros_like(aligned_target_angles)
        interpolated_velocities = np.zeros_like(aligned_target_velocities)

        # 전체 궤적 기반 보간으로 변경
        # 3차 다항식 가중치 (부드러운 변화)
        t = np.linspace(0, 1, n_points)
        weights = t * t * (3 - 2 * t)  # 더 부드러운 전환 곡선
        
        # 각도 보간
        for i in range(n_points):
            # 처음 3개 관절은 쿼터니언 보간 (회전 보존)
            target_rad = np.radians(aligned_target_angles[i, :3])
            subject_rad = np.radians(aligned_subject_angles[i, :3])
            
            q_target = R.from_euler('xyz', target_rad)
            q_subject = R.from_euler('xyz', subject_rad)
            
            q_target_arr = q_target.as_quat()
            q_subject_arr = q_subject.as_quat()
            
            dot = np.sum(q_target_arr * q_subject_arr)
            if dot < 0:
                q_subject_arr = -q_subject_arr
                dot = -dot
                
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
                
            q_interp = R.from_quat(result)
            euler_angles = q_interp.as_euler('xyz', degrees=True)
            interpolated_degrees[i, :3] = euler_angles
            
            # 4번째 관절 보간 (클리핑 제거)
            target_val = aligned_target_angles[i, 3]
            subject_val = aligned_subject_angles[i, 3]
            interpolated_degrees[i, 3] = (1 - weights[i]) * target_val + weights[i] * subject_val
            
            # 각속도 보간 (에르미트 보간으로 부드러운 속도 전환)
            for j in range(4):
                v0 = aligned_target_velocities[i, j]
                v1 = aligned_subject_velocities[i, j]
                w = weights[i]
                h00 = 2*w**3 - 3*w**2 + 1
                h10 = w**3 - 2*w**2 + w
                h01 = -2*w**3 + 3*w**2
                h11 = w**3 - w**2
                interpolated_velocities[i, j] = h00*v0 + h10*0 + h01*v1 + h11*0

        # 모델 입력 데이터를 각도 + 각속도로 결합
        model_input = np.column_stack([interpolated_degrees, interpolated_velocities])

        # 모델 기반 관절 간 상호작용 적용
        with torch.no_grad():
            segments = []
            segment_size = 100
            
            for i in range(0, n_points, segment_size):
                segment = model_input[i:i+segment_size]
                if len(segment) == 0:
                    continue
                segment_tensor = torch.FloatTensor(segment).unsqueeze(0).to(self.device)
                joint_interactions = self.model(segment_tensor).squeeze(0).cpu().numpy()
                segments.append(joint_interactions)
            
            if segments:
                model_output = np.vstack(segments)
                if len(model_output) > len(interpolated_degrees):
                    model_output = model_output[:len(interpolated_degrees)]
                
                # 모델 영향력 적용 (클리핑 없음)
                correction_strength = 0.2
                for joint in range(4):
                    interpolated_degrees[:, joint] = (1 - correction_strength) * interpolated_degrees[:, joint] + correction_strength * model_output[:, joint]
        
        # 확장된 허용 범위 클리핑 적용
        for joint in range(4):
            for i in range(len(interpolated_degrees)):
                if i < len(aligned_target_angles) and i < len(aligned_subject_angles):
                    # 타겟과 사용자 궤적 사이의 범위 계산
                    lower_bound = min(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                    upper_bound = max(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                    
                    # 범위 확장 (30% 여유)
                    range_width = upper_bound - lower_bound
                    if range_width > 0:
                        extended_lower = lower_bound - range_width * 0.3
                        extended_upper = upper_bound + range_width * 0.3
                    else:
                        # 두 값이 같은 경우 기본 여유 제공
                        extended_lower = lower_bound - 2
                        extended_upper = upper_bound + 2
                    
                    # 확장된 범위와 물리적 관절 한계 중 더 제한적인 범위 선택
                    extended_lower = max(extended_lower, self.joint_limits[joint][0])
                    extended_upper = min(extended_upper, self.joint_limits[joint][1])
                    
                    # 확장된 범위를 벗어나면 클리핑
                    if interpolated_degrees[i, joint] < extended_lower or interpolated_degrees[i, joint] > extended_upper:
                        interpolated_degrees[i, joint] = np.clip(interpolated_degrees[i, joint], extended_lower, extended_upper)

        # 강화된 글로벌 스무딩 적용
        # from scipy.signal import savgol_filter
        # window_length = min(101, n_points // 2 * 2 + 1)  # 훨씬 더 큰 윈도우 사용
        # window_length = max(5, window_length)  # 최소 5
        
        # for joint in range(4):
        #     # 매우 강한 스무딩 적용
        #     interpolated_degrees[:, joint] = savgol_filter(
        #         interpolated_degrees[:, joint], 
        #         window_length, 
        #         3,  # 3차 다항식
        #         mode='interp'  # 끝부분 외삽법 처리
        #     )
        
        # 각속도 재계산
        for joint in range(4):
            interpolated_velocities[:, joint] = np.gradient(interpolated_degrees[:, joint])
        
        return interpolated_degrees, interpolated_velocities
        
    def interpolate_trajectory(self, target_df, user_df, trajectory_type, weights=None):
        """궤적 타입에 따른 모델 기반 보간 수행 (가중치 None으로 동적 생성)"""
        # 각속도 계산 및 DataFrame 생성
        target_with_vel = target_df.copy()
        user_with_vel = user_df.copy()
        
        # 각속도 계산 추가
        for df in [target_with_vel, user_with_vel]:
            df['degsec1'] = np.gradient(df['deg1'])
            df['degsec2'] = np.gradient(df['deg2'])
            df['degsec3'] = np.gradient(df['deg3'])
            df['degsec4'] = np.gradient(df['deg4'])
        
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
        
        # 가중치가 None이면 0.2, 0.4, 0.6, 0.8로 동적 생성
        if weights is None:
            weights = [0.2, 0.4, 0.6, 0.8]
        
        results = {}
        # 보간 방법 선택 및 적용
        if 'clock' in trajectory_type.lower() or 'counter' in trajectory_type.lower():
            print("Using circular interpolation")
            for w in weights:
                degrees, velocities = self.model_based_interpolate_circle(target_data, user_data, interpolation_weight=w)
                results[f"weight_{w}"] = (degrees, velocities)
        elif 'v_' in trajectory_type.lower() or 'h_' in trajectory_type.lower():
            print("Using circular arc interpolation")
            for w in weights:
                degrees, velocities = self.model_based_interpolate_arc(target_data, user_data, interpolation_weight=w)
                results[f"weight_{w}"] = (degrees, velocities)
        else:
            print("Using linear interpolation")
            for w in weights:
                degrees, velocities = self.model_based_interpolate_line(target_data, user_data, interpolation_weight=w)
                results[f"weight_{w}"] = (degrees, velocities)
        
        # 결과 처리 (가장 마지막 가중치 결과만 DataFrame으로 변환)
        final_degrees = results[f"weight_{max(weights)}"][0]
        final_velocities = results[f"weight_{max(weights)}"][1]
        
        generated_df = pd.DataFrame(
            np.column_stack([final_degrees]),
            columns=['deg1', 'deg2', 'deg3', 'deg4']
        )
        
        generated_df['degsec1'] = final_velocities[:, 0]
        generated_df['degsec2'] = final_velocities[:, 1]
        generated_df['degsec3'] = final_velocities[:, 2]
        generated_df['degsec4'] = final_velocities[:, 3]
        
        generated_smoothed = self.smooth_data(generated_df)
        
        # for joint in ['deg1', 'deg2', 'deg3', 'deg4']:
            # for i in range(len(generated_smoothed)):
            #     if i < len(target_df) and i < len(user_df):
            #         # lower_bound = min(target_df[joint].iloc[i], user_df[joint].iloc[i])
            #         # upper_bound = max(target_df[joint].iloc[i], user_df[joint].iloc[i])
            #         if generated_smoothed[joint].iloc[i] < lower_bound or generated_smoothed[joint].iloc[i] > upper_bound:
            #             generated_smoothed.at[i, joint] = np.clip(generated_smoothed[joint].iloc[i], lower_bound, upper_bound)
        
        smoothed_degrees = generated_smoothed[['deg1', 'deg2', 'deg3', 'deg4']].values
        endeffector_degrees = smoothed_degrees.copy()
        # endeffector_degrees[:, 1] -= 90
        # endeffector_degrees[:, 3] -= 90

        aligned_points = np.array([calculate_end_effector_position(deg) for deg in endeffector_degrees])
        aligned_points = aligned_points * 1000
        
        final_df = pd.DataFrame(
            np.column_stack([aligned_points, smoothed_degrees]),
            columns=['x_end', 'y_end', 'z_end', 'deg1', 'deg2', 'deg3', 'deg4']
        )
        
        return final_df, results

    def visualize_trajectories(self, target_df, user_df, generated_df, trajectory_type, generation_number=1):
        """타겟과 사용자 궤적과 생성된 궤적을 시각화"""
        target_degrees = target_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_degrees = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        generated_degrees = generated_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        
        target_degrees_adj = target_degrees.copy()
        # target_degrees_adj[:, 1] -= 90
        # target_degrees_adj[:, 3] -= 90
        
        user_degrees_adj = user_degrees.copy()
        # user_degrees_adj[:, 1] -= 90
        # user_degrees_adj[:, 3] -= 90
        
        target_ends = np.array([calculate_end_effector_position(deg) for deg in target_degrees_adj]) * 1000
        user_ends = np.array([calculate_end_effector_position(deg) for deg in user_degrees_adj]) * 1000

        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(2, 3, figure=fig)
        
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

            # for i in range(min(len(target_time), len(user_time))):
            #     if i < len(target_time) and i < len(user_time):
            #         lower = min(target_df[joint].iloc[i], user_df[joint].iloc[i])
            #         upper = max(target_df[joint].iloc[i], user_df[joint].iloc[i])
            #         ax.fill_between([i, i+1], [lower, lower], [upper, upper], color='gray', alpha=0.2)
            
            # for i in range(len(generated_df)):
            #     if i < len(target_df) and i < len(user_df):
            #         lower = min(target_df[joint].iloc[i], user_df[joint].iloc[i])
            #         upper = max(target_df[joint].iloc[i], user_df[joint].iloc[i])
            #         val = generated_df[joint].iloc[i]
            #         if val < lower or val > upper:
            #             ax.plot(i, val, 'rx', markersize=8)

        fig.suptitle(f'Trajectory_type: {trajectory_type}', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

        self.save_generated_trajectory(generated_df, trajectory_type, generation_number)
        
        return generated_df
    
    def save_generated_trajectory(self, generated_df, classification_result, generation_number=1):
        """생성된 궤적을 지정된 형식으로 저장"""
        generation_dir = os.path.join(os.getcwd(), "generation_trajectory")
        os.makedirs(generation_dir, exist_ok=True)
        
        filename = f"generation_trajectory_{classification_result}_{generation_number}.txt"
        generation_path = os.path.join(generation_dir, filename)
        
        num_points = len(generated_df)
        full_df = pd.DataFrame(index=range(num_points))
        
        full_df['r'] = 'm'
        full_df['sequence'] = range(num_points)
        full_df['timestamp'] = [i * 10 for i in range(num_points)]
        
        full_df['deg'] = (generated_df['deg1'].round(3).astype(str) + '/' + 
                        generated_df['deg2'].round(3).astype(str) + '/' +
                        generated_df['deg3'].round(3).astype(str) + '/' +
                        generated_df['deg4'].round(3).astype(str))

        full_df['endpoint'] = (generated_df['x_end'].round(3).astype(str) + '/' + 
                            generated_df['y_end'].round(3).astype(str) + '/' + 
                            generated_df['z_end'].round(3).astype(str))
        
        full_df = full_df[['r', 'sequence', 'timestamp', 'deg', 'endpoint']]
        
        full_df.to_csv(generation_path, index=False)
        print(f"\nCompleted saving the generated trajectory: {generation_path}")

        return generation_path
        
    # def analyze_joint_relationships(self):
    #     """관절 간 상관관계 분석 및 시각화"""
    #     with torch.no_grad():
    #         dummy_input = torch.eye(4).unsqueeze(0).to(self.device)
    #         dummy_input = self.model.joint_embedding(dummy_input)
            
    #         attentions = []
    #         for layer in self.model.joint_attention_layers:
    #             Q = layer.query(dummy_input)
    #             K = layer.key(dummy_input)
    #             scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(layer.d_model).float())
    #             attention = torch.softmax(scores, dim=-1)
    #             attentions.append(attention.squeeze(0).cpu().numpy())
        
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
            
    #         for j in range(4):
    #             for k in range(4):
    #                 text = ax.text(k, j, f'{attn[j, k]:.2f}',
    #                               ha="center", va="center", color="w" if attn[j, k] > 0.5 else "black")
            
    #         fig.colorbar(im, ax=ax)
        
    #     plt.tight_layout()
    #     plt.suptitle('Interarticular Correlation Analysis', fontsize=16)
    #     plt.subplots_adjust(top=0.85)
    #     plt.show()
        
    #     avg_attention = np.mean(np.array(attentions), axis=0)
    #     np.fill_diagonal(avg_attention, 0) 
        
    #     strongest_pairs = []
    #     for i in range(4):
    #         for j in range(i+1, 4):
    #             strongest_pairs.append((i, j, avg_attention[i, j]))
        
    #     strongest_pairs.sort(key=lambda x: x[2], reverse=True)
        
    #     print("\nStrongest Joint Correlation (Top 3):")
    #     for i, (joint1, joint2, strength) in enumerate(strongest_pairs[:3]):
    #         print(f"{i+1}. {joint_names[joint1]} ↔ {joint_names[joint2]}: {strength:.4f}")