import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.interpolate import splprep, splev, UnivariateSpline
from utils import *
from analyzer import TrajectoryAnalyzer

class DegreeTransition:
    def __init__(self, l1=0.26, l2=0.31, k=0.01):
        self.l1 = l1
        self.l2 = l2
        self.k = k
        
        # 관절 값 제한
        self.joint_limits =  {
            0: (-10, 110),  
            1: (0, 150),    
            2: (0, 150),   
            3: (-90, 90)   
        }
        
    def calculate_end_effector_velocity(self, trajectory_data):
        """ 엔드이펙터 속도 계산 """
        # 관절 각도를 라디안으로 변환 (deg1, deg2, deg3, deg4)
        q1_rad = np.deg2rad(trajectory_data['deg1'].values)
        q2_rad = np.deg2rad(trajectory_data['deg2'].values)
        q3_rad = np.deg2rad(trajectory_data['deg3'].values)
        q4_rad = np.deg2rad(trajectory_data['deg4'].values)

        # 관절 각속도를 라디안/초로 변환 (degsec1, degsec2, degsec3, degsec4)
        qdot1_rad = np.deg2rad(trajectory_data['degsec1'].values)
        qdot2_rad = np.deg2rad(trajectory_data['degsec2'].values)
        qdot3_rad = np.deg2rad(trajectory_data['degsec3'].values)
        qdot4_rad = np.deg2rad(trajectory_data['degsec4'].values)

        # 결과를 저장할 배열 초기화
        num_samples = len(trajectory_data)
        ee_velocities = np.zeros((num_samples, 6))

        # 자코비안 행렬 계산
        for i in range(num_samples):
             J = calculate_jacobian_np(q1_rad[i], q2_rad[i], q3_rad[i], q4_rad[i])
             qdot = np.array([qdot1_rad[i], qdot2_rad[i], qdot3_rad[i], qdot4_rad[i]])

             # 엔드이펙터 속도 계산
             ee_velocities[i] = J @ qdot

        vel_df = pd.DataFrame(ee_velocities, columns=['vx', 'vy', 'vz', 'ωx', 'ωy', 'ωz'])

        return vel_df
    
    def calculate_joint_velocities(self, ee_velocity, trajectory_data):
        """ 관절 각속도 계산 """
        # 각도 값 가져오기
        deg1 = trajectory_data['deg1']
        deg2 = trajectory_data['deg2'] 
        deg3 = trajectory_data['deg3']
        deg4 = trajectory_data['deg4']

        # 자코비안 역행렬 계산
        J_inv = calculate_jacobian_inverse(deg1, deg2, deg3, deg4)

        # 관절 각속도 변환
        joint_velocities = np.dot(J_inv, ee_velocity)

        # 결과를 도/초 단위로 변환 (필요한 경우)
        # joint_velocities_deg = np.degrees(joint_velocities)
        
        return joint_velocities

    def next_joint_values(self, current_joint_values, joint_velocities, delta_t):
        # 현재 관절 값과 속도를 NumPy 배열로 변환
        if isinstance(current_joint_values, pd.Series):
            current_joint_values = current_joint_values.values
        if isinstance(joint_velocities, pd.Series):
            joint_velocities = joint_velocities.values
        
        # 다음 관절 값 계산: 현재 값 + 속도 × 시간
        next_joint_values = current_joint_values + joint_velocities * delta_t
        
        # 관절 각도 제한 적용 (필요한 경우)
        joint_limits = {
            0: (-10, 110),  # deg1 제한
            1: (0, 150),    # deg2 제한
            2: (0, 150),    # deg3 제한
            3: (-90, 90)    # deg4 제한
        }
        
        for i, (lower, upper) in joint_limits.items():
            if i < len(next_joint_values):
                next_joint_values[i] = np.clip(next_joint_values[i], lower, upper)
        
        return next_joint_values

    def generate_trajectory_caclulate_degrees(self, ee_path, timestamps, initial_joint_values, max_iterations=100, tolerance=0.001):
        """ 생성된 궤적의 엔드이펙터마다 각 관절 값 구하기 """
        # 결과를 저장할 리스트 초기화
        joint_trajectory = []
        current_joint_values = np.array(initial_joint_values)
        joint_trajectory.append(current_joint_values.copy())
        
        # 이전 타임스탬프 초기화
        prev_timestamp = timestamps[0]
        
        # 각 엔드이펙터 위치에 대해 반복 (첫 번째 지점은 이미 처리됨)
        for i in range(1, len(ee_path)):
            # 현재 타임스탬프와 delta_t 계산
            current_timestamp = timestamps[i]
            delta_t = current_timestamp - prev_timestamp
            
            # 엔드이펙터 속도 가져오기
            ee_velocity = ee_path[i]
            
            # 라디안으로 변환 (자코비안 계산용)
            current_joint_rad = np.deg2rad(current_joint_values)
            
            # 자코비안 역행렬 계산
            J_inv = calculate_jacobian_inverse(
                current_joint_values[0],
                current_joint_values[1],
                current_joint_values[2],
                current_joint_values[3]
            )
            
            # 관절 각속도 계산
            joint_velocities = np.dot(J_inv, ee_velocity)
            
            # 도/초 단위로 변환
            joint_velocities_deg = np.degrees(joint_velocities)
            
            # 다음 관절 값 예측
            next_joint_values = self.predict_next_joint_values(
                current_joint_values, 
                joint_velocities_deg, 
                delta_t
            )
            
            # 관절 값 업데이트
            current_joint_values = next_joint_values
            joint_trajectory.append(current_joint_values.copy())
            
            # 타임스탬프 업데이트
            prev_timestamp = current_timestamp
        
        # 리스트를 DataFrame으로 변환
        joint_df = pd.DataFrame(joint_trajectory, columns=['deg1', 'deg2', 'deg3', 'deg4'])
        
        # 타임스탬프 열 추가
        joint_df['timestamp'] = timestamps
        
        return joint_df

class EndeffectorInterpolate:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 관절 제한 설정
        self.joint_limits = {
            0: (-10, 110),
            1: (0, 150),     
            2: (0, 150),     
            3: (-90, 90)  
        }

    def smooth_data(self, data, smoothing_factor=100, density_factor=1.0, num_points=None):
        """ 궤적 스무딩 """
        # 원본 데이터 복사
        smoothed_data = data.copy()
        smoothed_endpoint = calculate_end_effector_position(smoothed_data['deg1', 'deg2', 'deg3', 'deg4'])

        # 시간 인덱스 생성
        t = np.linspace(0, 1, len(smoothed_endpoint[0]))

        # 시퀀스 길이 결정
        if num_points is None:
            num_points = int(len(smoothed_endpoint[0]) * density_factor)

        # 3차원 스플라인 피팅
        tck, u = splprep([smoothed_endpoint[0], smoothed_endpoint[1], smoothed_endpoint[2]], s=smoothing_factor, k=3)

        # 출력될 시퀀스 길이를 기반으로 분포 재설정
        new_t = np.linspace(0, 1, num_points)
        x_new, y_new, z_new = splev(new_t, tck)
        
        return np.column_stack((x_new, y_new, z_new))
    
    def normalize_time(self, target_data, user_data, num_points = None):
        """ 시간 정규화 """
        target_degrees = target_data[['deg1', 'deg2', 'deg3', 'deg4']].values
        subject_degrees = user_data[['deg1', 'deg2', 'deg3', 'deg4']].values

        # 엔드 이펙터 위치 계산
        target_endpoint = np.array([calculate_end_effector_position(deg) for deg in target_degrees])
        user_endpoint = np.array([calculate_end_effector_position(deg) for deg in subject_degrees])

        if num_points is None:
            num_points = max(len(target_data), len(user_data))

        # DTW를 사용한 정렬
        _, path = fastdtw(target_endpoint, user_endpoint, dist=euclidean)
        path = np.array(path)    

        # 정렬된 궤적 생성
        aligned_target = np.array([target_endpoint[i] for i in path[:, 0]])
        aligned_user = np.array([user_endpoint[j] for j in path[:, 1]])

        # 균일한 길이로 리샘플링
        def resample(points, n):
            t = np.linspace(0, 1, len(points))
            t_new = np.linspace(0, 1, n)
            points = np.array(points)
            x = UnivariateSpline(t, points[:, 0], s=0)(t_new)
            y = UnivariateSpline(t, points[:, 1], s=0)(t_new)
            z = UnivariateSpline(t, points[:, 2], s=0)(t_new)
            return np.column_stack([x, y, z])

        nomalized_target = resample(aligned_target, num_points)
        normalized__user = resample(aligned_user, num_points)

        return nomalized_target, normalized__user
    
    def linear_interpolate(self, target, user, interpolate_weight=0.5):
        """ 대각선 궤적 선형 보간 """
        # 대상과 사용자 데이터에서 관절 각도 추출
        target_degrees = target[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_degrees = user[['deg1', 'deg2', 'deg3', 'deg4']].values
        
        # 엔드이펙터 위치 계산
        target_endpoint = np.array([calculate_end_effector_position(deg) for deg in target_degrees])
        user_endpoint = np.array([calculate_end_effector_position(deg) for deg in user_degrees])
        
        # DTW를 사용하여 시간 정규화 수행
        target_normalized, user_normalized = self.nomazlize_time(target, user)
        
        # 두 궤적 사이의 선형 보간 수행
        num_points = len(target_normalized)
        interpolated_ee = np.zeros_like(target_normalized)
        
        # 선형 보간 수행
        for i in range(num_points):
            interpolated_ee[i] = (1 - interpolate_weight) * target_normalized[i] + interpolate_weight * user_normalized[i]
        
        # 타임스탬프 생성 (균등 간격)
        timestamps = np.linspace(0, 1, num_points)
        
        # 초기 관절 각도 (첫 번째 프레임에서 가져옴)
        initial_joint_values = target['deg1'].iloc[0], target['deg2'].iloc[0], target['deg3'].iloc[0], target['deg4'].iloc[0]
        
        # 역운동학을 통해 보간된 엔드이펙터 위치에 해당하는 관절 각도 계산
        degree_transition = DegreeTransition()  # DegreeTransition 클래스의 인스턴스 생성
        joint_trajectory = degree_transition.generate_trajectory_caclulate_degrees(
            interpolated_ee, timestamps, initial_joint_values
        )
        
        return joint_trajectory

    def arc_interpolate(self, target, user, interpolate_weight = 0.5):
        """ 호 궤적 보간 """
        # 대상과 사용자 데이터에서 관절 각도 추출
        target_degrees = target[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_degrees = user[['deg1', 'deg2', 'deg3', 'deg4']].values
        
        # 엔드이펙터 위치 계산
        target_endpoint = np.array([calculate_end_effector_position(deg) for deg in target_degrees])
        user_endpoint = np.array([calculate_end_effector_position(deg) for deg in user_degrees])
        
        # 호 길이 매개변수화
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
            
            # 엄격하게 증가하는 수열로 만들기
            for i in range(1, len(params)):
                if params[i] <= params[i-1]:
                    params[i] = params[i-1] + 1e-10
            
            return params, arc_lengths[-1]
        
        # 타겟과 사용자 궤적의 호 길이 매개변수화
        target_params, target_length = arc_length_parameterization(target_endpoint)
        user_params, user_length = arc_length_parameterization(user_endpoint)
        
        # 균일한 매개변수 간격으로 샘플링할 포인트 개수
        num_points = max(len(target_degrees), len(user_degrees))
        uniform_params = np.linspace(0, 1, num_points)
        
        # 스플라인을 사용하여 균일한 매개변수에서 위치 보간
        from scipy.interpolate import CubicSpline
        
        # 대상 궤적의 스플라인 인터폴레이션
        target_spline_x = CubicSpline(target_params, target_endpoint[:, 0], bc_type='natural')
        target_spline_y = CubicSpline(target_params, target_endpoint[:, 1], bc_type='natural')
        target_spline_z = CubicSpline(target_params, target_endpoint[:, 2], bc_type='natural')
        
        # 사용자 궤적의 스플라인 인터폴레이션
        user_spline_x = CubicSpline(user_params, user_endpoint[:, 0], bc_type='natural')
        user_spline_y = CubicSpline(user_params, user_endpoint[:, 1], bc_type='natural')
        user_spline_z = CubicSpline(user_params, user_endpoint[:, 2], bc_type='natural')
        
        # 균일한 매개변수에서 대상과 사용자 위치 계산
        target_uniform = np.zeros((num_points, 3))
        user_uniform = np.zeros((num_points, 3))
        
        for i, t in enumerate(uniform_params):
            target_uniform[i, 0] = target_spline_x(t)
            target_uniform[i, 1] = target_spline_y(t)
            target_uniform[i, 2] = target_spline_z(t)
            
            user_uniform[i, 0] = user_spline_x(t)
            user_uniform[i, 1] = user_spline_y(t)
            user_uniform[i, 2] = user_spline_z(t)
        
        # 곡률 기반 제어점 추출 (중요 변곡점 식별)
        def compute_curvature(points):
            n = len(points)
            curvature = np.zeros(n)
            
            # 첫 번째와 마지막 점 제외
            for i in range(1, n-1):
                # 전방 차분과 후방 차분
                v1 = points[i] - points[i-1]
                v2 = points[i+1] - points[i]
                
                # 속도 벡터 크기
                speed1 = np.linalg.norm(v1)
                speed2 = np.linalg.norm(v2)
                
                if speed1 < 1e-10 or speed2 < 1e-10:
                    curvature[i] = 0
                    continue
                
                # 단위 벡터
                t1 = v1 / speed1
                t2 = v2 / speed2
                
                # 곡률은 탄젠트 벡터의 변화율에 근사
                dt = t2 - t1
                curvature[i] = np.linalg.norm(dt) / ((speed1 + speed2) / 2)
            
            # 경계값 처리
            curvature[0] = curvature[1]
            curvature[-1] = curvature[-2]
            
            return curvature
        
        # 대상과 사용자 궤적의 곡률 계산
        target_curvature = compute_curvature(target_uniform)
        user_curvature = compute_curvature(user_uniform)
        
        # 곡률이 높은 지점을 제어점으로 선택 (상위 20% 포인트)
        combined_curvature = np.maximum(target_curvature, user_curvature)
        curvature_threshold = np.percentile(combined_curvature, 80)
        control_indices = list(np.where(combined_curvature > curvature_threshold)[0])
        
        # 시작점과 끝점 추가
        if 0 not in control_indices:
            control_indices.insert(0, 0)
        if (num_points - 1) not in control_indices:
            control_indices.append(num_points - 1)
        
        # 최소 제어점 수 보장 (최소 5개)
        if len(control_indices) < 5:
            additional_needed = 5 - len(control_indices)
            step = num_points // (additional_needed + 1)
            for i in range(1, additional_needed + 1):
                idx = i * step
                if idx not in control_indices and idx < num_points - 1:
                    control_indices.append(idx)
        
        # 제어점 정렬
        control_indices.sort()
        
        # 균일한 시간 매개변수
        t = np.linspace(0, 1, num_points)
        t_control = np.array([t[i] for i in control_indices])
        
        # 제어점에서의 보간 값 계산
        control_points = np.zeros((len(control_indices), 3))
        for i, idx in enumerate(control_indices):
            # 가중치 기반 보간
            control_points[i] = (1 - interpolate_weight) * target_uniform[idx] + interpolate_weight * user_uniform[idx]
        
        # 최종 보간 곡선 생성 (자연 스플라인)
        final_spline = CubicSpline(t_control, control_points, bc_type='natural')
        interpolated_ee = final_spline(t)
        
        # 타임스탬프 생성
        timestamps = t
        
        # 초기 관절 각도 (첫 번째 프레임에서 가져옴)
        initial_joint_values = target['deg1'].iloc[0], target['deg2'].iloc[0], target['deg3'].iloc[0], target['deg4'].iloc[0]
        
        # 역운동학을 통해 보간된 엔드이펙터 위치에 해당하는 관절 각도 계산
        degree_transition = DegreeTransition()
        joint_trajectory = degree_transition.generate_trajectory_caclulate_degrees(
            interpolated_ee, timestamps, initial_joint_values
        )
        
        return joint_trajectory
    
    def circle_interpolate(self, target, user, interpolate_weight = 0.5):
        """ 원 궤적 보간 """
        # 대상과 사용자 데이터에서 관절 각도 추출
        target_degrees = target[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_degrees = user[['deg1', 'deg2', 'deg3', 'deg4']].values
        
        # 엔드이펙터 위치 계산
        target_endpoint = np.array([calculate_end_effector_position(deg) for deg in target_degrees])
        user_endpoint = np.array([calculate_end_effector_position(deg) for deg in user_degrees])
        
        # 원형 궤적의 중심점 찾기
        target_center = np.mean(target_endpoint, axis=0)
        user_center = np.mean(user_endpoint, axis=0)
        
        # 보간된 중심점
        interpolated_center = (1 - interpolate_weight) * target_center + interpolate_weight * user_center
        
        # 중심까지의 벡터 계산
        target_centered = target_endpoint - target_center
        user_centered = user_endpoint - user_center
        
        # 위상각 계산 (xy 평면에서)
        target_phases = np.arctan2(target_centered[:, 1], target_centered[:, 0])
        user_phases = np.arctan2(user_centered[:, 1], user_centered[:, 0])
        
        # 각도 연속성 보장 (언래핑)
        target_phases = np.unwrap(target_phases)
        user_phases = np.unwrap(user_phases)
        
        # 위상 정규화 (0~1 범위로)
        target_phase_norm = (target_phases - target_phases.min()) / (target_phases.max() - target_phases.min())
        user_phase_norm = (user_phases - user_phases.min()) / (user_phases.max() - user_phases.min())
        
        # 공통 위상 포인트 생성
        num_points = max(len(target_degrees), len(user_degrees))
        common_phases = np.linspace(0, 1, num_points)
        
        # 각 위상에 해당하는 인덱스 찾기
        def find_nearest_indices(phases, common_phases):
            indices = np.zeros(len(common_phases), dtype=int)
            for i, phase in enumerate(common_phases):
                indices[i] = np.argmin(np.abs(phases - phase))
            return indices
        
        target_indices = find_nearest_indices(target_phase_norm, common_phases)
        user_indices = find_nearest_indices(user_phase_norm, common_phases)
        
        # 각 위상에서의 반경 계산
        target_radii = np.zeros(num_points)
        user_radii = np.zeros(num_points)
        target_heights = np.zeros(num_points)
        user_heights = np.zeros(num_points)
        
        for i in range(num_points):
            if i < len(target_indices):
                idx = target_indices[i]
                if idx < len(target_centered):
                    # XY 평면에서의 반경
                    target_radii[i] = np.sqrt(target_centered[idx, 0]**2 + target_centered[idx, 1]**2)
                    # Z 높이
                    target_heights[i] = target_centered[idx, 2]
            
            if i < len(user_indices):
                idx = user_indices[i]
                if idx < len(user_centered):
                    # XY 평면에서의 반경
                    user_radii[i] = np.sqrt(user_centered[idx, 0]**2 + user_centered[idx, 1]**2)
                    # Z 높이
                    user_heights[i] = user_centered[idx, 2]
        
        # 반경과 높이 보간
        interpolated_radii = (1 - interpolate_weight) * target_radii + interpolate_weight * user_radii
        interpolated_heights = (1 - interpolate_weight) * target_heights + interpolate_weight * user_heights
        
        # 회전 속도 보간 (시작과 끝 위상의 차이)
        target_rotation_range = target_phases.max() - target_phases.min()
        user_rotation_range = user_phases.max() - user_phases.min()
        
        # 회전 방향 보존 (시계/반시계)
        interpolated_rotation_range = (1 - interpolate_weight) * target_rotation_range + interpolate_weight * user_rotation_range
        
        # 최종 위상각 생성
        interpolated_phases = np.linspace(0, interpolated_rotation_range, num_points)
        
        # 보간된 엔드이펙터 궤적 생성
        interpolated_ee = np.zeros((num_points, 3))
        for i in range(num_points):
            phase = interpolated_phases[i]
            radius = interpolated_radii[i]
            
            # 극좌표를 데카르트 좌표로 변환
            interpolated_ee[i, 0] = interpolated_center[0] + radius * np.cos(phase)
            interpolated_ee[i, 1] = interpolated_center[1] + radius * np.sin(phase)
            interpolated_ee[i, 2] = interpolated_center[2] + interpolated_heights[i]
        
        # 타임스탬프 생성
        timestamps = np.linspace(0, 1, num_points)
        
        # 초기 관절 각도 (첫 번째 프레임에서 가져옴)
        initial_joint_values = target['deg1'].iloc[0], target['deg2'].iloc[0], target['deg3'].iloc[0], target['deg4'].iloc[0]
        
        # 역운동학을 통해 보간된 엔드이펙터 위치에 해당하는 관절 각도 계산
        degree_transition = DegreeTransition()
        joint_trajectory = degree_transition.generate_trajectory_caclulate_degrees(
            interpolated_ee, timestamps, initial_joint_values
        )
        
        return joint_trajectory
    
    def interpolate_trajectory(generator, target_df, user_df, trajectory_type, weights=None):
        """ 궤적 유형에 따라 적절한 보간 방법을 선택하여 궤적을 생성 """
        print(f"보간 방식: {trajectory_type} 유형 궤적 보간 수행 중...")
        
        # 가중치가 None이면 기본값 설정
        if weights is None:
            weights = [0.2, 0.4, 0.6, 0.8]
        
        # 각속도 계산 및 추가
        target_with_vel = target_df.copy()
        user_with_vel = user_df.copy()
        
        for df in [target_with_vel, user_with_vel]:
            if 'degsec1' not in df.columns:
                df['degsec1'] = np.gradient(df['deg1'])
                df['degsec2'] = np.gradient(df['deg2'])
                df['degsec3'] = np.gradient(df['deg3'])
                df['degsec4'] = np.gradient(df['deg4'])
        
        # 결과 저장 사전
        results = {}
        
        # 보간 방법 선택 및 각 가중치에 대한 보간 수행
        for w in weights:
            print(f"가중치 {w}로 보간 중...")
            
            # 궤적 유형에 따라 적절한 보간 방법 선택
            if any(t in trajectory_type.lower() for t in ['linear', 'straight', 'line']):
                print("선형 궤적 보간 적용")
                joint_trajectory = generator.linear_interpolate(target_with_vel, user_with_vel, w)
            elif any(t in trajectory_type.lower() for t in ['clock', 'counter']):
                print("원형 궤적 보간 적용")
                joint_trajectory = generator.circle_interpolate(target_with_vel, user_with_vel, w)
            elif any(t in trajectory_type.lower() for t in ['h_', 'v_', 'arc', 'curve']):
                print("호 궤적 보간 적용")
                joint_trajectory = generator.arc_interpolate(target_with_vel, user_with_vel, w)
            else:
                print(f"알 수 없는 궤적 유형: {trajectory_type}, 선형 보간 적용")
                joint_trajectory = generator.linear_interpolate(target_with_vel, user_with_vel, w)
            
            # 결과 저장
            results[f"weight_{w}"] = joint_trajectory
        
        # 가장 높은 가중치의 결과를 반환
        final_df = results[f"weight_{max(weights)}"]
        
        # 엔드이펙터 위치 계산 및 추가
        degrees = final_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        endpoints = np.array([calculate_end_effector_position(deg) for deg in degrees]) * 1000
        
        final_df['x_end'] = endpoints[:, 0]
        final_df['y_end'] = endpoints[:, 1] 
        final_df['z_end'] = endpoints[:, 2]
        
        return final_df, results
    


            

