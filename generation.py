import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from fastdtw import fastdtw
from utils import calculate_end_effector_position
from generation_model import JointTrajectoryTransformer
from scipy.spatial.distance import euclidean
from scipy.spatial.transform import Rotation as R   
from scipy.interpolate import UnivariateSpline, CubicSpline, PchipInterpolator
from scipy.signal import find_peaks

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

    def smooth_data(self, data, smoothing_factor=0.5, degree=3):
        """ 스플라인 스무딩을 사용하여 로봇 관절 각도와 각속도 데이터 스무딩 """
        # 원본 데이터 복사
        smoothed_df = data.copy()
        
        # 각 열에 대해 스플라인 스무딩 적용
        for col in ['deg1', 'deg2', 'deg3', 'deg4', 'degsec1', 'degsec2', 'degsec3', 'degsec4']:
            if col in data.columns:
                # 시간 인덱스 생성
                x = np.arange(len(data))
                y = data[col].values
                
                # 데이터의 규모에 따라 스무딩 매개변수 조정
                s = smoothing_factor * len(data)
                spline = UnivariateSpline(x, y, k=degree, s=s)
                smoothed_df[col] = spline(x)
                
                # 각속도 열을 스무딩할 때 물리적 제약 고려
                if col.startswith('degsec'):
                    angle_col = 'deg' + col[6:]
                    if angle_col in data.columns:
                        pass
        
        return smoothed_df
            
    def normalize_time(self, target_trajectory, subject_trajectory):
        """궤적 유형에 따라 적절한 정규화 방법 호출"""
        if hasattr(self, 'current_type'):
            if 'clock' in self.current_type.lower() or 'counter' in self.current_type.lower():
                return self.normalize_time_circle(target_trajectory, subject_trajectory)
            elif 'v_' in self.current_type.lower() or 'h_' in self.current_type.lower():
                return self.normalize_time_arc(target_trajectory, subject_trajectory)
        
        # 기본 정규화 방법 (일반 선형 궤적용)
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
        # 기본 샘플 수 설정 (더 긴 궤적에 맞춤)
        if num_points is None:
            num_points = max(len(target_trajectory), len(subject_trajectory))
        
        # 각도와 각속도 분리
        target_angles = target_trajectory[:, :4]
        target_velocities = target_trajectory[:, 4:]
        subject_angles = subject_trajectory[:, :4]
        subject_velocities = subject_trajectory[:, 4:]
        
        # 엔드이펙터 위치 계산
        target_ee_positions = np.zeros((len(target_angles), 3))
        subject_ee_positions = np.zeros((len(subject_angles), 3))
        
        for i in range(len(target_angles)):
            angles_adj = target_angles[i].copy()
            target_ee_positions[i] = calculate_end_effector_position(angles_adj)
        
        for i in range(len(subject_angles)):
            angles_adj = subject_angles[i].copy()
            subject_ee_positions[i] = calculate_end_effector_position(angles_adj)
        
        # 호 길이 기반 매개변수화
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
        
        # 균일한 매개변수 간격으로 샘플링할 포인트
        uniform_params = np.linspace(0, 1, num_points)
        
        # 재샘플링된 궤적을 저장할 배열
        resampled_target = np.zeros((num_points, target_trajectory.shape[1]))
        resampled_subject = np.zeros((num_points, subject_trajectory.shape[1]))
        
        for dim in range(target_trajectory.shape[1]):
            # 타겟 궤적 스플라인 피팅
            target_spline = CubicSpline(
                target_params, 
                target_trajectory[:, dim],
                bc_type='natural'
            )
            resampled_target[:, dim] = target_spline(uniform_params)
            
            # 사용자 궤적 스플라인 피팅
            subject_spline = CubicSpline(
                subject_params, 
                subject_trajectory[:, dim],
                bc_type='natural'
            )
            resampled_subject[:, dim] = subject_spline(uniform_params)
        
        # 클리핑 완화
        for joint in range(4):
            joint_min = self.joint_limits[joint][0]
            joint_max = self.joint_limits[joint][1]
            
            # 물리적 한계 내로만 클리핑
            resampled_target[:, joint] = np.clip(resampled_target[:, joint], joint_min, joint_max)
            resampled_subject[:, joint] = np.clip(resampled_subject[:, joint], joint_min, joint_max)
        
        return resampled_target, resampled_subject

    def normalize_time_circle(self, target_trajectory, subject_trajectory):
        """원형 궤적 특화 시간 정규화 (위상 기반)"""
        num_points = max(len(target_trajectory), len(subject_trajectory))
        
        # 각도 데이터
        target_angles = target_trajectory[:, :4]
        subject_angles = subject_trajectory[:, :4]
        
        # 엔드이펙터 위치 계산
        target_ee_positions = np.zeros((len(target_angles), 3))
        subject_ee_positions = np.zeros((len(subject_angles), 3))
        
        for i in range(len(target_angles)):
            target_ee_positions[i] = calculate_end_effector_position(target_angles[i])
        
        for i in range(len(subject_angles)):
            subject_ee_positions[i] = calculate_end_effector_position(subject_angles[i])
        
        # 중심점 계산
        target_center = np.mean(target_ee_positions, axis=0)
        subject_center = np.mean(subject_ee_positions, axis=0)
        
        # 각 위치에서 중심까지의 벡터 계산
        target_centered = target_ee_positions - target_center
        subject_centered = subject_ee_positions - subject_center
        
        # 위상 각도 계산 (xy 평면에서)
        target_phases = np.arctan2(target_centered[:, 1], target_centered[:, 0])
        subject_phases = np.arctan2(subject_centered[:, 1], subject_centered[:, 0])
        
        # 각도 연속성 보장
        target_phases = np.unwrap(target_phases)
        subject_phases = np.unwrap(subject_phases)
        
        # 위상 정규화 (0~1 범위로)
        if len(target_phases) > 1:
            target_phase_norm = (target_phases - target_phases.min()) / (target_phases.max() - target_phases.min())
        else:
            target_phase_norm = np.array([0])
            
        if len(subject_phases) > 1:
            subject_phase_norm = (subject_phases - subject_phases.min()) / (subject_phases.max() - subject_phases.min())
        else:
            subject_phase_norm = np.array([0])
        
        # 공통 위상 포인트 생성
        common_phases = np.linspace(0, 1, num_points)
        
        # 균일한 위상 간격에서의 인덱스 찾기
        target_indices = np.zeros(num_points, dtype=int)
        subject_indices = np.zeros(num_points, dtype=int)
        
        for i, phase in enumerate(common_phases):
            target_idx = np.argmin(np.abs(target_phase_norm - phase))
            subject_idx = np.argmin(np.abs(subject_phase_norm - phase))
            
            target_indices[i] = target_idx
            subject_indices[i] = subject_idx
        
        # 정규화된 궤적 생성
        resampled_target = target_trajectory[target_indices]
        resampled_subject = subject_trajectory[subject_indices]
        
        return resampled_target, resampled_subject

    def normalize_time(self, target_trajectory, subject_trajectory):
        """각 관절의 특징점을 기반으로 시간 정규화"""
        # 결과 배열
        num_points = max(len(target_trajectory), len(subject_trajectory))
        resampled_target = np.zeros((num_points, target_trajectory.shape[1]))
        resampled_subject = np.zeros((num_points, subject_trajectory.shape[1]))
        
        # 각도 데이터
        target_angles = target_trajectory[:, :4]
        subject_angles = subject_trajectory[:, :4]
        
        # 각 관절을 독립적으로 처리
        for joint in range(4):
            # 1. 각도 unwrap
            target_joint = np.unwrap(target_angles[:, joint] * np.pi/180) * 180/np.pi
            subject_joint = np.unwrap(subject_angles[:, joint] * np.pi/180) * 180/np.pi
            
            # 피크 찾기
            target_peaks, _ = find_peaks(target_joint, prominence=2)
            target_valleys, _ = find_peaks(-target_joint, prominence=2)
            subject_peaks, _ = find_peaks(subject_joint, prominence=2)
            subject_valleys, _ = find_peaks(-subject_joint, prominence=2)
            
            # 변곡점 찾기
            target_diff = np.gradient(target_joint)
            subject_diff = np.gradient(subject_joint)
            target_infl = np.where(np.diff(np.signbit(np.gradient(target_diff))))[0]
            subject_infl = np.where(np.diff(np.signbit(np.gradient(subject_diff))))[0]
            
            # 특징점 결합 및 정렬
            target_features = np.sort(np.concatenate([[0], target_peaks, target_valleys, target_infl, [len(target_joint)-1]]))
            subject_features = np.sort(np.concatenate([[0], subject_peaks, subject_valleys, subject_infl, [len(subject_joint)-1]]))
            
            # 중복 제거
            target_features = np.unique(target_features)
            subject_features = np.unique(subject_features)
            
            # 특징점 매칭
            min_count = min(len(target_features), len(subject_features))
            if min_count < 3: 
                target_params = np.linspace(0, 1, len(target_joint))
                subject_params = np.linspace(0, 1, len(subject_joint))
            else:
                if len(target_features) > min_count:
                    indices = np.round(np.linspace(0, len(target_features)-1, min_count)).astype(int)
                    target_features = target_features[indices]
                
                if len(subject_features) > min_count:
                    indices = np.round(np.linspace(0, len(subject_features)-1, min_count)).astype(int)
                    subject_features = subject_features[indices]
                
                target_params = target_features / (len(target_joint) - 1)
                subject_params = subject_features / (len(subject_joint) - 1)
            
            # 공통 시간축
            common_time = np.linspace(0, 1, num_points)
            
            # 매핑 함수
            time_map = PchipInterpolator(subject_params, target_params, extrapolate=True)
            mapped_time = common_time  # 여기서는 직접 매핑하지 않고 다음 단계에서 사용
            
            # 6. 보간 적용
            target_time = np.linspace(0, 1, len(target_joint))
            subject_time = np.linspace(0, 1, len(subject_joint))
            
            # 타겟 보간
            target_interp = PchipInterpolator(target_time, target_joint, extrapolate=True)
            resampled_target[:, joint] = target_interp(common_time)
            
            # 사용자 데이터를 타겟에 맞춰 보간
            # 여기서 중요한 변경: 특징점 기반 매핑 적용
            mapped_subject_time = time_map(subject_time)
            subject_interp = PchipInterpolator(mapped_subject_time, subject_joint, extrapolate=True)
            resampled_subject[:, joint] = subject_interp(common_time)
        
        # 각속도 처리 (관절과 유사하게)
        target_velocities = target_trajectory[:, 4:]
        subject_velocities = subject_trajectory[:, 4:]
        
        for vel_idx in range(target_velocities.shape[1]):
            # 간단한 선형 매핑 사용
            target_time = np.linspace(0, 1, len(target_velocities))
            subject_time = np.linspace(0, 1, len(subject_velocities))
            
            target_interp = PchipInterpolator(target_time, target_velocities[:, vel_idx], extrapolate=True)
            subject_interp = PchipInterpolator(subject_time, subject_velocities[:, vel_idx], extrapolate=True)
            
            resampled_target[:, 4 + vel_idx] = target_interp(common_time)
            resampled_subject[:, 4 + vel_idx] = subject_interp(common_time)
        
        return resampled_target, resampled_subject

    def model_based_interpolate_line(self, target, subject, interpolation_weight=None):
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
                
                correction_strength = 0.5
                for joint in range(4):
                    for i in range(len(interpolated_degrees)):
                        original_val = interpolated_degrees[i, joint]
                        interpolated_degrees[i, joint] = (1 - correction_strength) * original_val + correction_strength * model_output[i, joint]

        # 각속도 재계산
        for joint in range(4):
            interpolated_velocities[:, joint] = np.gradient(interpolated_degrees[:, joint])

        return interpolated_degrees, interpolated_velocities

    def model_based_interpolate_arc(self, target, subject, interpolation_weight=None):
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
            control_indices = [0, n_points//4, n_points//2, 3*n_points//4, n_points-1]
            control_times = [0, 0.25, 0.5, 0.75, 1]
            control_angles = []
            control_velocities = []
            
            # 제어점 값 계산
            for ci in control_indices:
                target_val = aligned_target_angles[ci, joint]
                subject_val = aligned_subject_angles[ci, joint]
                
                # 가중치 보간
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
                
                correction_strength = 0.5
                for joint in range(4):
                    interpolated_degrees[:, joint] = (1 - correction_strength) * interpolated_degrees[:, joint] + correction_strength * model_output[:, joint]
        
        # 확장된 허용 범위 클리핑 적용
        for joint in range(4):
            for i in range(len(interpolated_degrees)):
                if i < len(aligned_target_angles) and i < len(aligned_subject_angles):
                    # 타겟과 사용자 궤적 사이의 범위 계산
                    lower_bound = min(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                    upper_bound = max(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                    range_width = upper_bound - lower_bound
                    if range_width > 0:
                        extended_lower = lower_bound - range_width * 0.3
                        extended_upper = upper_bound + range_width * 0.3
                    else:
                        extended_lower = lower_bound - 2
                        extended_upper = upper_bound + 2
                    
                    # 확장된 범위와 물리적 관절 한계 중 더 제한적인 범위 선택
                    extended_lower = max(extended_lower, self.joint_limits[joint][0])
                    extended_upper = min(extended_upper, self.joint_limits[joint][1])
                    
                    # 확장된 범위를 벗어나면 클리핑
                    if interpolated_degrees[i, joint] < extended_lower or interpolated_degrees[i, joint] > extended_upper:
                        interpolated_degrees[i, joint] = np.clip(interpolated_degrees[i, joint], extended_lower, extended_upper)
        
        # 각속도 재계산
        for joint in range(4):
            interpolated_velocities[:, joint] = np.gradient(interpolated_degrees[:, joint])

        return interpolated_degrees, interpolated_velocities


    def model_based_interpolate_circle(self, target, subject, interpolation_weight=None):
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

        # 3차 다항식 가중치
        t = np.linspace(0, 1, n_points)
        weights = t * t * (3 - 2 * t) 
        
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
            
            # 4번째 관절 보간
            target_val = aligned_target_angles[i, 3]
            subject_val = aligned_subject_angles[i, 3]
            interpolated_degrees[i, 3] = (1 - weights[i]) * target_val + weights[i] * subject_val
            
            # 각속도 보간
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
                
                correction_strength = 0.5
                for joint in range(4):
                    interpolated_degrees[:, joint] = (1 - correction_strength) * interpolated_degrees[:, joint] + correction_strength * model_output[:, joint]
        
        # 확장된 허용 범위 클리핑 적용
        for joint in range(4):
            for i in range(len(interpolated_degrees)):
                if i < len(aligned_target_angles) and i < len(aligned_subject_angles):
                    # 타겟과 사용자 궤적 사이의 범위 계산
                    lower_bound = min(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                    upper_bound = max(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                    range_width = upper_bound - lower_bound
                    if range_width > 0:
                        extended_lower = lower_bound - range_width * 0.3
                        extended_upper = upper_bound + range_width * 0.3
                    else:
                        extended_lower = lower_bound - 2
                        extended_upper = upper_bound + 2
                
                    extended_lower = max(extended_lower, self.joint_limits[joint][0])
                    extended_upper = min(extended_upper, self.joint_limits[joint][1])
                    
                    if interpolated_degrees[i, joint] < extended_lower or interpolated_degrees[i, joint] > extended_upper:
                        interpolated_degrees[i, joint] = np.clip(interpolated_degrees[i, joint], extended_lower, extended_upper)
        
        # 각속도 재계산
        for joint in range(4):
            interpolated_velocities[:, joint] = np.gradient(interpolated_degrees[:, joint])
        
        return interpolated_degrees, interpolated_velocities
        
    def interpolate_trajectory(self, target_df, user_df, trajectory_type, weights=None):
        """궤적 타입에 따른 모델 기반 보간 수행 (가중치 None으로 동적 생성)"""
        # 각속도 계산 및 DataFrame 생성
        self.current_type = trajectory_type
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
        
        smoothed_degrees = generated_smoothed[['deg1', 'deg2', 'deg3', 'deg4']].values
        endeffector_degrees = smoothed_degrees.copy()

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
        user_degrees_adj = user_degrees.copy()
        
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