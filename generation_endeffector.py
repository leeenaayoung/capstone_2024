import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sympy as sp
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.interpolate import splprep, splev, UnivariateSpline
from utils import *
from analyzer import TrajectoryAnalyzer
from endeffector_model import TrajectoryTransformer

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
    
    def load_transformer_model(self, model_path="best_trajectory_transformer.pth"):
        """학습된 Transformer 모델 로드"""
        # 기기 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # 체크포인트 로드
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 기본 모델 파라미터 (모델 구조에 맞게 설정 필요)
            input_dim = 7  # 위치(3) + 각도(4)
            self.transformer = TrajectoryTransformer(
                input_dim=input_dim, 
                d_model=256, 
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6
            )
            
            # 모델 가중치 로드
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.transformer.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.transformer.load_state_dict(checkpoint)
                
            # 평가 모드로 설정
            self.transformer.to(self.device)
            self.transformer.eval()
            
            print(f"Transformer 모델이 성공적으로 로드되었습니다: {model_path}")
            return True
        
        except Exception as e:
            print(f"Transformer 모델 로드 중 오류 발생: {str(e)}")
            return False
    
    # def interpolate_trajectory(self, target_df, user_df, trajectory_type, weights=0.5):
    #     """ 가중치 기반 궤적 보간 - 직접 공간 보간과 각도 보간 모두 수행 """
    #     # 공간상의 엔드이펙터 보간
    #     normalized_target, normalized_user = self.normalize_time(target_df, user_df)
    #     interpolated_points = (1 - weights) * normalized_user + weights * normalized_target
        
    #     # 관절 각도 보간
    #     interpolated_degrees = self.interpolate_degrees(target_df, user_df, weights)

    #     num_points = len(interpolated_points)
    #     interpolated_df = pd.DataFrame({
    #         'x_end': interpolated_points[:, 0],
    #         'y_end': interpolated_points[:, 1],
    #         'z_end': interpolated_points[:, 2],
    #         'deg1': interpolated_degrees[:, 0],
    #         'deg2': interpolated_degrees[:, 1],
    #         'deg3': interpolated_degrees[:, 2],
    #         'deg4': interpolated_degrees[:, 3],
    #         'timestamps': np.linspace(0, 1, num_points)
    #     })
        
    #     # 결과 정보
    #     results = {
    #         'trajectory_type': trajectory_type,
    #         'interpolation_weight': weights,
    #         'num_points': num_points,
    #         'space_error': np.mean(np.linalg.norm(normalized_target - normalized_user, axis=1)),
    #         'joint_error': np.mean(np.abs(
    #             target_df[['deg1', 'deg2', 'deg3', 'deg4']].values.mean(axis=0) - 
    #             user_df[['deg1', 'deg2', 'deg3', 'deg4']].values.mean(axis=0)
    #         ))
    #     }

    #     interpolated_df = self.smooth_data(interpolated_df, smoothing_factor=0.5)
        
    #     return interpolated_df, results
    
    def _prepare_data_for_transformer(self, df):
        """데이터프레임을 Transformer 입력 형식으로 변환"""
        # 관절 각도 추출
        angles = df[['deg1', 'deg2', 'deg3', 'deg4']].values
        
        # 엔드이펙터 위치 계산 또는 추출
        if 'x_end' in df.columns:
            positions = df[['x_end', 'y_end', 'z_end']].values
        else:
            positions = np.array([calculate_end_effector_position(deg) for deg in angles]) * 1000
        
        # 특성 결합 (위치 + 각도)
        features = np.concatenate([positions, angles], axis=1)
        
        # 텐서로 변환
        return torch.tensor(features, dtype=torch.float32).to(self.device)

    def _compute_spatial_encoding(self, target_df, user_df):
        """궤적의 공간적 특성을 인코딩"""
        # 타겟 궤적의 공간적 특성
        target_angles = target_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        target_ee = np.array([calculate_end_effector_position(deg) for deg in target_angles]) * 1000
        
        # 사용자 궤적의 공간적 특성
        user_angles = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_ee = np.array([calculate_end_effector_position(deg) for deg in user_angles]) * 1000
        
        # 공간적 특성 계산
        # 시작점, 끝점, 중심점
        target_start = target_ee[0]
        target_end = target_ee[-1]
        target_center = np.mean(target_ee, axis=0)
        
        user_start = user_ee[0]
        user_end = user_ee[-1]
        user_center = np.mean(user_ee, axis=0)
        
        # 방향 벡터
        target_dir = target_end - target_start
        target_dir_norm = np.linalg.norm(target_dir)
        if target_dir_norm > 0:
            target_dir = target_dir / target_dir_norm
        
        user_dir = user_end - user_start
        user_dir_norm = np.linalg.norm(user_dir)
        if user_dir_norm > 0:
            user_dir = user_dir / user_dir_norm
        
        # 공간 크기
        target_size = np.linalg.norm(np.max(target_ee, axis=0) - np.min(target_ee, axis=0))
        user_size = np.linalg.norm(np.max(user_ee, axis=0) - np.min(user_ee, axis=0))
        
        # 16차원 공간 인코딩 생성
        spatial_encoding = np.concatenate([
            target_start,      # 타겟 시작점 (3)
            target_end,        # 타겟 끝점 (3)
            target_center,     # 타겟 중심점 (3)
            user_start,        # 사용자 시작점 (3)
            user_end,          # 사용자 끝점 (3)
            user_center,       # 사용자 중심점 (3)
            [target_size],     # 타겟 크기 (1)
            [user_size]        # 사용자 크기 (1)
        ])
        
        return torch.tensor(spatial_encoding, dtype=torch.float32).unsqueeze(0).to(self.device)

    def interpolate_trajectory(self, target_df, user_df, trajectory_type, weights=0.5):
        """가중치 기반 궤적 보간 - 직접 공간 보간과 각도 보간에 Transformer 적용"""
        # Transformer 모델 로드 (아직 로드되지 않은 경우)
        if not hasattr(self, 'transformer'):
            self.load_transformer_model()
        
        # 기존 방식으로 시간 정규화 및 보간
        normalized_target, normalized_user = self.normalize_time(target_df, user_df)
        interpolated_points_basic = (1 - weights) * normalized_user + weights * normalized_target
        
        # 관절 각도 보간 (기존 방식)
        interpolated_degrees = self.interpolate_degrees(target_df, user_df, weights)
        
        # Transformer가 성공적으로 로드된 경우, 위치 정보 향상
        if hasattr(self, 'transformer'):
            # 데이터 준비
            target_data = self._prepare_data_for_transformer(target_df)
            user_data = self._prepare_data_for_transformer(user_df)
            
            # 궤적 타입 인코딩
            type_mapping = {"d_": 0, "clock": 1, "counter": 2, "v_": 3, "h_": 4}
            type_idx = type_mapping.get(trajectory_type, 0)
            type_tensor = torch.tensor([type_idx], device=self.device)
            
            try:
                # Transformer를 사용한 궤적 보간 - 인자 수정
                with torch.no_grad():
                    enhanced_trajectory = self.transformer.interpolate_trajectories(
                        trajectory1=user_data,
                        trajectory2=target_data,
                        interpolate_weight=weights,
                        traj_type=type_tensor
                        # spatial_encoding 인자 제거됨
                    )
                    
                    # 결과를 numpy 배열로 변환
                    enhanced_positions = enhanced_trajectory[:, :3].cpu().numpy()
                    
                    # 길이 조정 (필요한 경우)
                    if len(enhanced_positions) != len(interpolated_points_basic):
                        from scipy.interpolate import interp1d
                        
                        t_orig = np.linspace(0, 1, len(enhanced_positions))
                        t_target = np.linspace(0, 1, len(interpolated_points_basic))
                        
                        # x, y, z 각각에 대해 보간
                        enhanced_resampled = np.zeros_like(interpolated_points_basic)
                        for i in range(3):  # x, y, z
                            interp_func = interp1d(t_orig, enhanced_positions[:, i], kind='cubic')
                            enhanced_resampled[:, i] = interp_func(t_target)
                        
                        enhanced_positions = enhanced_resampled
                    
                    # 기존 보간과 Transformer 결과 결합
                    alpha = 0.6  # Transformer 결과의 가중치 (0.0~1.0)
                    interpolated_points = (1 - alpha) * interpolated_points_basic + alpha * enhanced_positions
                
            except Exception as e:
                print(f"Transformer 보간 중 오류 발생: {e}")
                print("기존 보간 방식만 사용합니다.")
                interpolated_points = interpolated_points_basic
        else:
            # Transformer가 로드되지 않은 경우, 기존 방식만 사용
            interpolated_points = interpolated_points_basic
        
        # 데이터프레임 생성
        num_points = len(interpolated_points)
        interpolated_df = pd.DataFrame({
            'x_end': interpolated_points[:, 0],
            'y_end': interpolated_points[:, 1],
            'z_end': interpolated_points[:, 2],
            'deg1': interpolated_degrees[:, 0],
            'deg2': interpolated_degrees[:, 1],
            'deg3': interpolated_degrees[:, 2],
            'deg4': interpolated_degrees[:, 3],
            'timestamps': np.linspace(0, 1, num_points)
        })
        
        # 결과 정보
        results = {
            'trajectory_type': trajectory_type,
            'interpolation_weight': weights,
            'num_points': num_points,
            'space_error': np.mean(np.linalg.norm(normalized_target - normalized_user, axis=1)),
            'joint_error': np.mean(np.abs(
                target_df[['deg1', 'deg2', 'deg3', 'deg4']].values.mean(axis=0) - 
                user_df[['deg1', 'deg2', 'deg3', 'deg4']].values.mean(axis=0)
            )),
            'transformer_enhanced': hasattr(self, 'transformer')
        }
        
        # 스무딩 적용
        interpolated_df = self.smooth_data(interpolated_df, smoothing_factor=0.5)
        
        return interpolated_df, results

    def visualize_trajectories(self, target_ee, user_ee, interpolated_ee, weight, trajectory_type, save_path=None, show=True):
        """ 타겟, 사용자, 보간된 궤적 시각화 """
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2]) 
        
        ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
        
        # # 타겟 궤적의 엔드이펙터 좌표 추출
        # if 'x_end' in target_ee.columns:
        #     target_x = target_ee['x_end'].values
        #     target_y = target_ee['y_end'].values
        #     target_z = target_ee['z_end'].values
        # else:
        #     # deg 값으로부터 엔드이펙터 계산
        target_degrees = target_ee[['deg1', 'deg2', 'deg3', 'deg4']].values
        target_endpoints = np.array([calculate_end_effector_position(deg) for deg in target_degrees]) * 1000
        target_x, target_y, target_z = target_endpoints[:, 0], target_endpoints[:, 1], target_endpoints[:, 2]
        
        # 사용자 궤적의 엔드이펙터 좌표 추출
        # if 'x_end' in user_ee.columns:
        #     user_x = user_ee['x_end'].values
        #     user_y = user_ee['y_end'].values
        #     user_z = user_ee['z_end'].values
        # else:
        #     # deg 값으로부터 엔드이펙터 계산
        user_degrees = user_ee[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_endpoints = np.array([calculate_end_effector_position(deg) for deg in user_degrees]) * 1000
        user_x, user_y, user_z = user_endpoints[:, 0], user_endpoints[:, 1], user_endpoints[:, 2]
        
        # 보간된 궤적의 엔드이펙터 좌표 추출
        interp_x = interpolated_ee['x_end'].values
        interp_y = interpolated_ee['y_end'].values
        interp_z = interpolated_ee['z_end'].values
        
        # 궤적 그리기
        ax_3d.plot(target_x, target_y, target_z, 'b-', linewidth=2, label='target trajectory')
        ax_3d.plot(user_x, user_y, user_z, 'r-', linewidth=2, label='user trajectory')
        ax_3d.plot(interp_x, interp_y, interp_z, 'g-', linewidth=2, label='interpolated trajectory')
        
        # # 시작점과 끝점 표시
        # ax_3d.scatter(target_x[0], target_y[0], target_z[0], c='b', marker='o', s=100)
        # ax_3d.scatter(target_x[-1], target_y[-1], target_z[-1], c='b', marker='x', s=100)
        
        # ax_3d.scatter(user_x[0], user_y[0], user_z[0], c='r', marker='o', s=100)
        # ax_3d.scatter(user_x[-1], user_y[-1], user_z[-1], c='r', marker='x', s=100)
        
        # 3D 그래프 설정
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title('End-effector Trajectory')
        ax_3d.legend()
        
        # 오른쪽 영역을 2x2 서브플롯으로 분할
        gs_right = gs[0, 1].subgridspec(2, 2)
        
        # 시간축 생성
        t_target = np.linspace(0, 200, len(target_ee))  # 0-200 범위로 정규화
        t_user = np.linspace(0, 200, len(user_ee))
        t_interp = np.linspace(0, 200, len(interpolated_ee))
        
        # 관절 각도 그래프 (4개의 관절 각도를 2x2 서브플롯으로)
        joint_names = ['deg1', 'deg2', 'deg3', 'deg4']
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  # 2x2 그리드 위치
        
        for i, (joint_name, pos) in enumerate(zip(joint_names, positions)):
            # 각 관절을 위한 서브플롯 생성
            ax = fig.add_subplot(gs_right[pos])
            
            # 각 관절 데이터 플로팅
            if joint_name in target_ee.columns:
                ax.plot(t_target, target_ee[joint_name].values, 'b--', linewidth=1.5, label='target')
            
            if joint_name in user_ee.columns:
                ax.plot(t_user, user_ee[joint_name].values, 'r:', linewidth=1.5, label='user')
            
            if joint_name in interpolated_ee.columns:
                ax.plot(t_interp, interpolated_ee[joint_name].values, 'g-', linewidth=1.5, label='interpolate')
            
            # 그래프 설정
            ax.set_xlabel('timestamp')
            ax.set_ylabel('deg')
            ax.set_title(joint_name)
            ax.grid(True)
            ax.legend()
        
        # 전체 제목 추가
        plt.suptitle(f'Trajectory type: {trajectory_type}', fontsize=16)
        
        # 레이아웃 조정
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 상단 제목을 위한 공간 확보
        
        # 그래프 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 그래프 표시
        if show:
            plt.show()
        else:
            plt.close()
    
    def interpolate_degrees(self, target_df, user_df, weight):
        """
        관절 각도 직접 보간
        
        Parameters:
        - target_df: 타겟 궤적 데이터프레임
        - user_df: 사용자 궤적 데이터프레임
        - weight: 보간 가중치 (0: 사용자, 1: 타겟)
        
        Returns:
        - interpolated_degrees: 보간된 관절 각도 배열
        """
        # DTW를 사용해 시퀀스 길이 맞추기
        target_degrees = target_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_degrees = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        
        # DTW에 사용할 특징 (관절 각도 기반)
        _, path = fastdtw(target_degrees, user_degrees, dist=euclidean)
        path = np.array(path)
        
        # 정렬된 관절 각도 생성
        aligned_target = np.array([target_degrees[i] for i in path[:, 0]])
        aligned_user = np.array([user_degrees[j] for j in path[:, 1]])
        
        # 균일한 길이로 리샘플링
        num_points = max(len(target_df), len(user_df))
        t = np.linspace(0, 1, len(aligned_target))
        t_new = np.linspace(0, 1, num_points)
        
        # 각 관절에 대해 스플라인 보간
        resampled_target = np.zeros((num_points, 4))
        resampled_user = np.zeros((num_points, 4))
        
        for i in range(4):  # 4개 관절
            spline_target = UnivariateSpline(t, aligned_target[:, i], s=0)
            spline_user = UnivariateSpline(t, aligned_user[:, i], s=0)
            
            resampled_target[:, i] = spline_target(t_new)
            resampled_user[:, i] = spline_user(t_new)
        
        # 가중치 기반 보간
        interpolated_degrees = (1 - weight) * resampled_user + weight * resampled_target
        
        # 관절 제한 적용
        for joint_idx, (min_angle, max_angle) in self.joint_limits.items():
            interpolated_degrees[:, joint_idx] = np.clip(
                interpolated_degrees[:, joint_idx], 
                min_angle, 
                max_angle
            )
        
        return interpolated_degrees
    
def main():
    print("\n======== Trajectory Generation Mode ========")
    
    # 디렉토리 및 모델 경로 설정
    base_dir = "data"
    model_path = "best_trajectory_transformer.pth"
    
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

    generator.load_transformer_model(model_path=model_path)

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