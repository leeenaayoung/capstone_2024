import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.interpolate import splprep, splev, UnivariateSpline, CubicSpline
from utils import *
from analyzer import TrajectoryAnalyzer

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
    
    def smooth_data(self, data, smoothing_factor=0.5, degree=3):
        """궤적 스플라인 스무딩"""
        # 원본 데이터 복사
        smoothed_df = data.copy()

        # 관절 각도 추출
        degrees = data[['deg1', 'deg2', 'deg3', 'deg4']].values

        # 엔드이펙터 위치 계산
        endpoints = np.array([calculate_end_effector_position(deg) for deg in degrees])

        # 엔드이펙터 위치를 x, y, z로 분리
        x = endpoints[:, 0]
        y = endpoints[:, 1]
        z = endpoints[:, 2]
        t = np.arange(len(data))

        # 스플라인 스무딩 적용 (degree 사용)
        s = smoothing_factor * len(data)
        spline_x = UnivariateSpline(t, x, k=degree, s=s)
        spline_y = UnivariateSpline(t, y, k=degree, s=s)
        spline_z = UnivariateSpline(t, z, k=degree, s=s)

        # 스무딩된 엔드이펙터 위치
        smoothed_endpoints = np.column_stack([spline_x(t), spline_y(t), spline_z(t)])

        # 결과를 DataFrame에 저장
        smoothed_df['x'] = smoothed_endpoints[:, 0]
        smoothed_df['y'] = smoothed_endpoints[:, 1]
        smoothed_df['z'] = smoothed_endpoints[:, 2]

        return smoothed_df
    
    def normalize_time(self, target_data, user_data, num_points = None):
        """ 시간 정규화 """
        # 스무딩 적용
        target_smoothed = self.smooth_data(target_data, smoothing_factor=0.5, degree=3)
        user_smoothed = self.smooth_data(user_data, smoothing_factor=0.5, degree=3)

        target_degrees = target_smoothed[['deg1', 'deg2', 'deg3', 'deg4']].values
        subject_degrees = user_smoothed[['deg1', 'deg2', 'deg3', 'deg4']].values

        # 엔드 이펙터 위치 계산
        target_endpoint = np.array([calculate_end_effector_position(deg) for deg in target_degrees]) * 1000
        user_endpoint = np.array([calculate_end_effector_position(deg) for deg in subject_degrees]) * 1000

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
        """대각선 궤적 선형 보간"""
        # 스무딩 적용
        target_smoothed = self.smooth_data(target, smoothing_factor=0.5, degree=3)
        user_smoothed = self.smooth_data(user, smoothing_factor=0.5, degree=3)

        target_endpoint = target_smoothed[['x', 'y', 'z']].values
        user_endpoint = user_smoothed[['x', 'y', 'z']].values

        # 방향성 분석
        target_direction = target_endpoint[-1] - target_endpoint[0]
        user_direction = user_endpoint[-1] - user_endpoint[0]
        direction_match = np.dot(target_direction, user_direction) > 0

        if not direction_match:
            user_endpoint = np.flip(user_endpoint, axis=0)

        # 정규화 수행
        target_normalized, user_normalized = self.normalize_time(target, user)

        # 선형 보간
        num_points = len(target_normalized)
        interpolated_ee = np.zeros_like(target_normalized)
        for i in range(num_points):
            interpolated_ee[i] = (1 - interpolate_weight) * target_normalized[i] + interpolate_weight * user_normalized[i]

        return interpolated_ee

    def arc_interpolate(self, target, user, interpolate_weight=0.5):
        """호 궤적 보간"""
        # 스무딩 적용
        target_smoothed = self.smooth_data(target, smoothing_factor=0.5, degree=3)
        user_smoothed = self.smooth_data(user, smoothing_factor=0.5, degree=3)

        target_endpoint = target_smoothed[['x', 'y', 'z']].values
        user_endpoint = user_smoothed[['x', 'y', 'z']].values

        def arc_length_parameterization(positions):
            arc_lengths = np.zeros(len(positions))
            for i in range(1, len(positions)):
                arc_lengths[i] = arc_lengths[i-1] + np.linalg.norm(positions[i] - positions[i-1])
            params = arc_lengths / arc_lengths[-1] if arc_lengths[-1] > 0 else np.linspace(0, 1, len(positions))
            for i in range(1, len(params)):
                if params[i] <= params[i-1]:
                    params[i] = params[i-1] + 1e-10
            return params, arc_lengths[-1]

        target_params, target_length = arc_length_parameterization(target_endpoint)
        user_params, user_length = arc_length_parameterization(user_endpoint)

        num_points = max(len(target_endpoint), len(user_endpoint))
        uniform_params = np.linspace(0, 1, num_points)

        target_spline_x = CubicSpline(target_params, target_endpoint[:, 0], bc_type='natural')
        target_spline_y = CubicSpline(target_params, target_endpoint[:, 1], bc_type='natural')
        target_spline_z = CubicSpline(target_params, target_endpoint[:, 2], bc_type='natural')

        user_spline_x = CubicSpline(user_params, user_endpoint[:, 0], bc_type='natural')
        user_spline_y = CubicSpline(user_params, user_endpoint[:, 1], bc_type='natural')
        user_spline_z = CubicSpline(user_params, user_endpoint[:, 2], bc_type='natural')

        target_uniform = np.column_stack([target_spline_x(uniform_params),
                                         target_spline_y(uniform_params),
                                         target_spline_z(uniform_params)])
        user_uniform = np.column_stack([user_spline_x(uniform_params),
                                       user_spline_y(uniform_params),
                                       user_spline_z(uniform_params)])

        def compute_curvature(points):
            n = len(points)
            curvature = np.zeros(n)
            for i in range(1, n-1):
                v1 = points[i] - points[i-1]
                v2 = points[i+1] - points[i]
                speed1 = np.linalg.norm(v1)
                speed2 = np.linalg.norm(v2)
                if speed1 < 1e-10 or speed2 < 1e-10:
                    curvature[i] = 0
                    continue
                t1 = v1 / speed1
                t2 = v2 / speed2
                dt = t2 - t1
                curvature[i] = np.linalg.norm(dt) / ((speed1 + speed2) / 2)
            curvature[0] = curvature[1]
            curvature[-1] = curvature[-2]
            return curvature

        target_curvature = compute_curvature(target_uniform)
        user_curvature = compute_curvature(user_uniform)
        combined_curvature = np.maximum(target_curvature, user_curvature)
        curvature_threshold = np.percentile(combined_curvature, 80)
        control_indices = list(np.where(combined_curvature > curvature_threshold)[0])

        if 0 not in control_indices:
            control_indices.insert(0, 0)
        if (num_points - 1) not in control_indices:
            control_indices.append(num_points - 1)

        if len(control_indices) < 5:
            additional_needed = 5 - len(control_indices)
            step = num_points // (additional_needed + 1)
            for i in range(1, additional_needed + 1):
                idx = i * step
                if idx not in control_indices and idx < num_points - 1:
                    control_indices.append(idx)

        control_indices.sort()
        t = np.linspace(0, 1, num_points)
        t_control = np.array([t[i] for i in control_indices])

        control_points = np.zeros((len(control_indices), 3))
        for i, idx in enumerate(control_indices):
            control_points[i] = (1 - interpolate_weight) * target_uniform[idx] + interpolate_weight * user_uniform[idx]

        final_spline = CubicSpline(t_control, control_points, bc_type='natural')
        interpolated_ee = final_spline(t)

        return interpolated_ee

    def circle_interpolate(self, target, user, interpolate_weight=0.5):
        """원 궤적 보간"""
        # 스무딩 적용
        target_smoothed = self.smooth_data(target, smoothing_factor=0.5, degree=3)
        user_smoothed = self.smooth_data(user, smoothing_factor=0.5, degree=3)

        target_endpoint = target_smoothed[['x', 'y', 'z']].values
        user_endpoint = user_smoothed[['x', 'y', 'z']].values

        # 시간 정규화
        target_normalized, user_normalized = self.normalize_time(target, user, num_points=max(len(target_endpoint), len(user_endpoint)))

        target_center = np.mean(target_normalized, axis=0)
        user_center = np.mean(user_normalized, axis=0)
        interpolated_center = (1 - interpolate_weight) * target_center + interpolate_weight * user_center

        target_centered = target_normalized - target_center
        user_centered = user_normalized - user_center

        target_phases = np.unwrap(np.arctan2(target_centered[:, 1], target_centered[:, 0]))
        user_phases = np.unwrap(np.arctan2(user_centered[:, 1], user_centered[:, 0]))

        phase_offset = target_phases[0] - user_phases[0]
        user_phases += phase_offset

        target_direction = np.sign(target_phases[-1] - target_phases[0])
        user_direction = np.sign(user_phases[-1] - user_phases[0])
        if target_direction != user_direction:
            user_normalized = np.flip(user_normalized, axis=0)
            user_centered = user_normalized - user_center
            user_phases = np.unwrap(np.arctan2(user_centered[:, 1], user_centered[:, 0])) + phase_offset

        target_phase_norm = (target_phases - target_phases.min()) / (target_phases.max() - target_phases.min())
        user_phase_norm = (user_phases - user_phases.min()) / (user_phases.max() - user_phases.min())

        num_points = len(target_normalized)
        common_phases = np.linspace(0, 1, num_points)

        def find_nearest_indices(phases, common_phases):
            return np.array([np.argmin(np.abs(phases - phase)) for phase in common_phases])

        target_indices = find_nearest_indices(target_phase_norm, common_phases)
        user_indices = find_nearest_indices(user_phase_norm, common_phases)

        target_radii = np.sqrt(target_centered[target_indices, 0]**2 + target_centered[target_indices, 1]**2)
        user_radii = np.sqrt(user_centered[user_indices, 0]**2 + user_centered[user_indices, 1]**2)
        target_heights = target_centered[target_indices, 2]
        user_heights = user_centered[user_indices, 2]

        t = np.linspace(0, 1, num_points)
        radius_spline = CubicSpline(t, (1 - interpolate_weight) * target_radii + interpolate_weight * user_radii)
        height_spline = CubicSpline(t, (1 - interpolate_weight) * target_heights + interpolate_weight * user_heights)

        interpolated_radii = radius_spline(t)
        interpolated_heights = height_spline(t)

        target_rotation_range = target_phases.max() - target_phases.min()
        user_rotation_range = user_phases.max() - user_phases.min()
        interpolated_rotation_range = (1 - interpolate_weight) * target_rotation_range + interpolate_weight * user_rotation_range

        interpolated_phases = np.linspace(0, interpolated_rotation_range, num_points)

        interpolated_ee = np.zeros((num_points, 3))
        for i in range(num_points):
            phase = interpolated_phases[i]
            radius = interpolated_radii[i]
            interpolated_ee[i, 0] = interpolated_center[0] + radius * np.cos(phase)
            interpolated_ee[i, 1] = interpolated_center[1] + radius * np.sin(phase)
            interpolated_ee[i, 2] = interpolated_center[2] + interpolated_heights[i]

        return interpolated_ee
    
    def interpolate_trajectory(self, target_df, user_df, trajectory_type, weights=None):
        """ 궤적 유형에 따라 적절한 보간 방법을 선택하여 궤적을 생성 """
        print(f"보간 방식: {trajectory_type} 유형 궤적 보간 수행 중...")
        
        # 가중치가 None이면 기본값 설정
        if weights is None:
            weights = 0.5
        
        # 결과 저장 사전
        results = {}
        
        # 궤적 유형에 따라 적절한 보간 방법 선택
        if any(t in trajectory_type.lower() for t in ['d_']):
            print("선형 궤적 보간 적용")
            interpolated_ee = self.linear_interpolate(target_df, user_df, weights)
        elif any(t in trajectory_type.lower() for t in ['clock', 'counter']):
            print("원형 궤적 보간 적용")
            interpolated_ee = self.circle_interpolate(target_df, user_df, weights)
        elif any(t in trajectory_type.lower() for t in ['h_', 'v_']):
            print("호 궤적 보간 적용")
            interpolated_ee = self.arc_interpolate(target_df, user_df, weights)
        else:
            print(f"알 수 없는 궤적 유형: {trajectory_type}, 선형 보간 적용")
            interpolated_ee = self.linear_interpolate(target_df, user_df, weights)
        
        # 결과 저장
        results[f"weight_{weights}"] = interpolated_ee

        return interpolated_ee, results
    
    def visualize_trajectories(self, target_ee, user_ee, interpolated_ee, weight=0.5, trajectory_type=None, save_path=None, show=True):
        """ 보간된 궤적 시각화 """
        if isinstance(target_ee, pd.DataFrame):
            target_ee = target_ee.values
        if isinstance(user_ee, pd.DataFrame):
            user_ee = user_ee.values
        if isinstance(interpolated_ee, pd.DataFrame):
            interpolated_ee = interpolated_ee.values

        # 3D 그림 생성
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 타겟 궤적 그리기
        ax.plot(target_ee[:, 0], target_ee[:, 1], target_ee[:, 2], 
                color='blue', linewidth=2, label='Target Trajectory')
        
        # 사용자 궤적 그리기
        ax.plot(user_ee[:, 0], user_ee[:, 1], user_ee[:, 2], 
                color='red', linewidth=2, label='User Trajectory')
        
        # 보간된 궤적 그리기
        ax.plot(interpolated_ee[:, 0], interpolated_ee[:, 1], interpolated_ee[:, 2], 
                color='green', linewidth=3, linestyle='-', 
                label=f'Interpolated (w={weight})')
        
        # 그래프 설정
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title(f'{trajectory_type.capitalize()} Trajectory Interpolation (weight={weight})')
        
        # 범례 추가
        ax.legend(loc='best')
        
        # 좌표축 비율 동일하게 설정
        all_x = np.concatenate([target_ee[:, 0], user_ee[:, 0], interpolated_ee[:, 0]])
        all_y = np.concatenate([target_ee[:, 1], user_ee[:, 1], interpolated_ee[:, 1]])
        all_z = np.concatenate([target_ee[:, 2], user_ee[:, 2], interpolated_ee[:, 2]])
        
        x_range = all_x.max() - all_x.min()
        y_range = all_y.max() - all_y.min()
        z_range = all_z.max() - all_z.min()
        max_range = max(x_range, y_range, z_range) / 2.0
        
        mid_x = all_x.mean()
        mid_y = all_y.mean()
        mid_z = all_z.mean()
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # 그리드 추가
        ax.grid(True)
        
        # 결과 파일로 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"그림이 {save_path}에 저장되었습니다.")
        
        # 화면에 표시
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
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
    
def main():

    print("\n======== Trajectory Generation Mode ========")
    
    # 디렉토리 및 모델 경로 설정
    base_dir = "data"
    model_path = "best_generation_model.pth"
    
    # 모델 경로 확인
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return False
    
    # 분석기 초기화
    try:
        analyzer = TrajectoryAnalyzer(
            classification_model="best_classification_model.pth",
            base_dir=base_dir
        )
    except Exception as e:
        print(f"Error: Analyzer initialization failed: {str(e)}")
        return False
    
    # 생성기 초기화
    generator = EndeffectorInterpolate(analyzer)

    # 사용자 궤적 파일 선택
    non_golden_dir = os.path.join(base_dir, "non_golden_sample")
    non_golden_files = [f for f in os.listdir(non_golden_dir) if f.endswith('.txt')]
    
    if not non_golden_files:
        print("사용자 궤적 파일을 찾을 수 없습니다.")
        return
    
     # 랜덤하게 하나의 사용자 궤적 선택
    selected_file = random.choice(non_golden_files)
    print(f"선택된 사용자 궤적: {selected_file}")
    
    # 사용자 궤적 로드 및 분류
    user_path = os.path.join(non_golden_dir, selected_file)
    user_trajectory, trajectory_type = analyzer.load_user_trajectory(user_path)
    
    # 해당 타입의 타겟 궤적 찾기
    golden_dir = os.path.join(base_dir, "golden_sample")
    golden_files = [f for f in os.listdir(golden_dir) if trajectory_type in f and f.endswith('.txt')]
    
    if not golden_files:
        print(f"{trajectory_type} 타입의 타겟 궤적을 찾을 수 없습니다.")
        return
    
    # 타겟 궤적 로드
    target_file = golden_files[0]
    print(f"매칭된 타겟 궤적: {target_file}")
    target_path = os.path.join(golden_dir, target_file)
    target_trajectory, _ = analyzer.load_user_trajectory(target_path)

    print("\nCreating model-based trajectories...")
    generated_df, results = generator.interpolate_trajectory(target_df=target_trajectory, user_df=user_trajectory, trajectory_type=trajectory_type, weights=0.5)

    print("\nVisualizing and saving trajectories...")
    interpolated_ee, results = generator.interpolate_trajectory(
        target_df=target_trajectory, 
        user_df=user_trajectory, 
        trajectory_type=trajectory_type, 
        weights=0.5
    )

    # 보간된 엔드이펙터 위치 직접 전달
    generator.visualize_trajectories(
        target_ee=target_trajectory,
        user_ee=user_trajectory, 
        interpolated_ee=interpolated_ee,
        weight=0.5,
        trajectory_type=trajectory_type,
        save_path=None,
        show=True
)

    print("\nProcessing completed!")
    return True

if __name__ == "__main__":
    main()



            

