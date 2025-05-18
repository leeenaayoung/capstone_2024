import os
import datetime
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.interpolate import UnivariateSpline, CubicSpline
from scipy.signal import savgol_filter
from utils import calculate_end_effector_position
from analyzer import TrajectoryAnalyzer
# from endeffector_model import TrajectoryTransformer
from generate_model import TrajectoryTransformer

class EndeffectorGenerator:
    def __init__(self, analyzer, model_path="best_trajectory_transformer_250.pth"):
        """ 트랜스포머 기반 엔드이펙터 궤적 생성기 """
        self.analyzer = analyzer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 관절 제한 설정
        self.joint_limits = {
            0: (-10, 110),
            1: (0, 150),     
            2: (0, 150),     
            3: (-90, 90)  
        }

        # 평가 결과에 따른 가중치 적용용
        self.grade_to_weight = {
            1: 0.2,  # 1등급 
            2: 0.4,  # 2등급
            3: 0.6,  # 3등급
            4: 0.8   # 4등급 
        }
        # 모델 불러오기
        self.transformer = self.load_transformer_model(model_path)
        
    def load_transformer_model(self, model_path):
        """학습된 트랜스포머 모델 불러오기"""
        try:
            # 모델 초기화
            transformer = TrajectoryTransformer(
                                        input_dim=7,
                                        d_model=128,  # 더 큰 모델 차원
                                        nhead=4,      # 더 많은 어텐션 헤드
                                        num_layers=4, # 더 깊은 네트워크
                                        dropout=0.1
                                    )
            
            # 체크포인트 로드
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 모델 가중치 로드
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                transformer.load_state_dict(checkpoint['model_state_dict'])
                print(f"모델 가중치 로드 완료: epoch {checkpoint.get('epoch', 'unknown')}, loss {checkpoint.get('loss', 'unknown')}")
            else:
                transformer.load_state_dict(checkpoint)
                print(f"모델 가중치 로드 완료")
            
            # 평가 모드로 설정
            transformer.to(self.device)
            transformer.eval()
            
            print(f"트랜스포머 모델 로드 성공: {model_path}")
            return transformer
            
        except Exception as e:
            print(f"모델 로드 중 에러 발생: {str(e)}")
            return None
            
    def smooth_data(self, data, window_length=21, polyorder=2):
        """Savitzky-Golay 필터 기반 궤적 스무딩"""
        smoothed_df = data.copy()

        degrees = data[['deg1', 'deg2', 'deg3', 'deg4']].values
        endpoints = np.array([calculate_end_effector_position(deg) for deg in degrees])
        t = np.arange(len(data))

        # 필터를 적용할 수 있는 최소 길이 보장
        if len(t) < window_length:
            window_length = len(t) if len(t) % 2 == 1 else len(t) - 1
            if window_length < 3:
                return smoothed_df  # 너무 짧으면 smoothing 하지 않음

        # 엔드이펙터 위치에 필터 적용
        smoothed_x = savgol_filter(endpoints[:, 0], window_length, polyorder)
        smoothed_y = savgol_filter(endpoints[:, 1], window_length, polyorder)
        smoothed_z = savgol_filter(endpoints[:, 2], window_length, polyorder)

        smoothed_df['x'] = smoothed_x
        smoothed_df['y'] = smoothed_y
        smoothed_df['z'] = smoothed_z

        return smoothed_df
    
    def normalize_time(self, target_data, user_data, num_points=100):
        """시간 정규화 및 DTW 적용"""
        # 스무딩 적용
        target_smoothed = self.smooth_data(target_data, window_length=21, polyorder=2)
        user_smoothed = self.smooth_data(user_data, window_length=21, polyorder=2)
        
        # 관절 각도 추출
        target_degrees = target_smoothed[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_degrees = user_smoothed[['deg1', 'deg2', 'deg3', 'deg4']].values

        # 엔드 이펙터 위치 계산
        target_endpoint = np.array([calculate_end_effector_position(deg) for deg in target_degrees]) * 1000
        user_endpoint = np.array([calculate_end_effector_position(deg) for deg in user_degrees]) * 1000

        # DTW를 사용한 정렬
        _, path = fastdtw(target_endpoint, user_endpoint, dist=euclidean)
        path = np.array(path)    

        # 정렬된 궤적 생성
        aligned_target = np.array([target_endpoint[i] for i in path[:, 0]])
        aligned_user = np.array([user_endpoint[j] for j in path[:, 1]])

        def resample(data, t_old, t_new):
            resampled = np.zeros((len(t_new), data.shape[1]))
            for i in range(data.shape[1]):
                spline = CubicSpline(t_old, data[:, i], bc_type='clamped')
                resampled[:, i] = spline(t_new)
            return resampled

        normalized_target = resample(aligned_target, num_points)
        normalized_user = resample(aligned_user, num_points)

        return normalized_target, normalized_user
    
    def prepare_trajectory_input(self, target_df, user_df, max_seq_length=250):
        """트랜스포머 모델 입력을 위한 궤적 데이터 준비 (위치/각도 분리 정규화)"""

        def extract_and_combine(df):
            degrees = df[['deg1', 'deg2', 'deg3', 'deg4']].values
            endpoints = np.array([calculate_end_effector_position(deg) for deg in degrees]) * 1000
            return np.concatenate([endpoints, degrees], axis=1)

        target_combined = extract_and_combine(target_df)
        user_combined = extract_and_combine(user_df)

        t_target = np.linspace(0, 1, len(target_combined))
        t_user = np.linspace(0, 1, len(user_combined))
        t_new = np.linspace(0, 1, max_seq_length)

        def resample(data, t_old, t_new):
            resampled = np.zeros((len(t_new), data.shape[1]))
            for i in range(data.shape[1]):
                spline = UnivariateSpline(t_old, data[:, i], s=0)
                resampled[:, i] = spline(t_new)
            return resampled

        target_resampled = resample(target_combined, t_target, t_new)
        user_resampled = resample(user_combined, t_user, t_new)

        target_pos, target_deg = target_resampled[:, :3], target_resampled[:, 3:]
        user_pos, user_deg = user_resampled[:, :3], user_resampled[:, 3:]

        # 위치 정규화
        all_pos = np.vstack([target_pos, user_pos])
        pos_mean = np.mean(all_pos, axis=0)
        pos_std = np.std(all_pos, axis=0) + 1e-8
        target_pos_norm = (target_pos - pos_mean) / pos_std
        user_pos_norm = (user_pos - pos_mean) / pos_std

        # 각도 정규화
        all_deg = np.vstack([target_deg, user_deg])
        deg_mean = np.mean(all_deg, axis=0)
        deg_std = np.std(all_deg, axis=0) + 1e-8
        target_deg_norm = (target_deg - deg_mean) / deg_std
        user_deg_norm = (user_deg - deg_mean) / deg_std

        # 병합
        target_normalized = np.concatenate([target_pos_norm, target_deg_norm], axis=1)
        user_normalized = np.concatenate([user_pos_norm, user_deg_norm], axis=1)
        interpolated_gt = (target_normalized + user_normalized) / 2.0

        normalization_params = {
            'pos_mean': pos_mean,
            'pos_std': pos_std,
            'deg_mean': deg_mean,
            'deg_std': deg_std,
            'max_seq_length': max_seq_length
        }

        target_tensor = torch.FloatTensor(target_normalized).unsqueeze(0).to(self.device)
        user_tensor = torch.FloatTensor(user_normalized).unsqueeze(0).to(self.device)
        interpolated_tensor = torch.FloatTensor(interpolated_gt).unsqueeze(0).to(self.device)

        return user_tensor, target_tensor, interpolated_tensor, normalization_params
    
    def generate_trajectory(self, target_df, user_df, trajectory_type):
        """트랜스포머 모델을 사용하여 새로운 궤적 생성"""
        if self.transformer is None:
            print("트랜스포머 모델이 로드되지 않았습니다.")
            return None, None

        try:
            user_tensor, target_tensor, interpolated_tensor, norm_params = self.prepare_trajectory_input(
                target_df, user_df)

            with torch.no_grad():
                generated = self.transformer(user_tensor, target_tensor, interpolated_gt=None)

            generated_np = generated.squeeze(0).cpu().numpy()

            # 위치와 각도 분리 후 역정규화
            pos_mean = norm_params['pos_mean']
            pos_std = norm_params['pos_std']
            deg_mean = norm_params['deg_mean']
            deg_std = norm_params['deg_std']

            generated_pos = generated_np[:, :3] * pos_std + pos_mean
            generated_deg = generated_np[:, 3:] * deg_std + deg_mean

            for joint_idx, (min_angle, max_angle) in self.joint_limits.items():
                generated_deg[:, joint_idx] = np.clip(generated_deg[:, joint_idx], min_angle, max_angle)

            generated_df = pd.DataFrame({
                'x_end': generated_pos[:, 0],
                'y_end': generated_pos[:, 1],
                'z_end': generated_pos[:, 2],
                'deg1': generated_deg[:, 0],
                'deg2': generated_deg[:, 1],
                'deg3': generated_deg[:, 2],
                'deg4': generated_deg[:, 3],
                'timestamps': np.linspace(0, 1, len(generated_pos))
            })

            results = {
                'trajectory_type': trajectory_type,
                'num_points': len(generated_df),
                'transformer_used': True
            }

            # 스무딩 적용
            # generated_df = self.smooth_data(generated_df, smoothing_factor=0.5)
            generated_df = self.smooth_data(generated_df, window_length=11, polyorder=3)

            return generated_df, results

        except Exception as e:
            print(f"궤적 생성 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def save_trajectory_to_csv(self, generated_df, file_path=None, trajectory_type="unknown"):
        """ 생성된 궤적 데이터프레임 저장 """
        # 필요한 열이 있는지 확인
        required_columns = ['x_end', 'y_end', 'z_end', 'deg1', 'deg2', 'deg3', 'deg4']
        missing_columns = [col for col in required_columns if col not in generated_df.columns]
        
        # 누락된 열이 있으면 경고 메시지 출력
        if missing_columns:
            print(f"경고: 다음 열이 데이터프레임에 없습니다: {missing_columns}")
            
            # 누락된 열 추가 (0으로 채움)
            for col in missing_columns:
                generated_df[col] = 0.0
                print(f"- '{col}' 열을 0으로 채워 추가했습니다.")
        
        # 열 순서 정렬
        final_df = generated_df[required_columns].copy()
        
        # timestamps 열이 있으면 추가
        if 'timestamps' in generated_df.columns:
            final_df['timestamps'] = generated_df['timestamps']
        
        # 파일 경로가 제공되지 않은 경우 기본 경로 생성
        if file_path is None:

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"generated_trajectory_{trajectory_type}_{timestamp}.csv"
        
        # CSV 파일로 저장
        try:
            final_df.to_csv(file_path, index=False)
            print(f"궤적 데이터가 성공적으로 저장되었습니다: {file_path}")
            return file_path
        except Exception as e:
            print(f"파일 저장 중 오류가 발생했습니다: {str(e)}")
            return None
        
    def visualize_trajectories(self, target_df, user_df, generated_df, trajectory_type, save_path=None, show=True):
        """궤적 시각화 (타겟, 사용자, 생성된 궤적)"""
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2]) 
        
        # 3D 그래프 생성
        ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
        
        # 타겟 궤적의 엔드이펙터 좌표 
        target_degrees = target_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        target_endpoints = np.array([calculate_end_effector_position(deg) for deg in target_degrees]) * 1000
        target_x, target_y, target_z = target_endpoints[:, 0], target_endpoints[:, 1], target_endpoints[:, 2]
        
        # 사용자 궤적의 엔드이펙터 좌표
        user_degrees = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_endpoints = np.array([calculate_end_effector_position(deg) for deg in user_degrees]) * 1000
        user_x, user_y, user_z = user_endpoints[:, 0], user_endpoints[:, 1], user_endpoints[:, 2]
        
        # 생성된 궤적의 엔드이펙터 좌표
        gen_x = generated_df['x_end'].values 
        gen_y = generated_df['y_end'].values 
        gen_z = generated_df['z_end'].values 
        
        # 궤적 그리기
        ax_3d.plot(target_x, target_y, target_z, 'b-', linewidth=2, label='Target Trajectory')
        ax_3d.plot(user_x, user_y, user_z, 'r-', linewidth=2, label='User Trajectory')
        ax_3d.plot(gen_x, gen_y, gen_z, 'g-', linewidth=2, label='Generated Trajectory')
        
        # 3D 그래프 설정
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title('End-effector Trajectory')
        ax_3d.legend()
        
        # 오른쪽 영역을 2x2 서브플롯으로 분할
        gs_right = gs[0, 1].subgridspec(2, 2)
        
        # 시간축 생성
        t_target = np.linspace(0, 200, len(target_df))  
        t_user = np.linspace(0, 200, len(user_df))
        t_gen = np.linspace(0, 200, len(generated_df))
        
        # 관절 각도 그래프
        joint_names = ['deg1', 'deg2', 'deg3', 'deg4']
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)] 
        
        for i, (joint_name, pos) in enumerate(zip(joint_names, positions)):
            ax = fig.add_subplot(gs_right[pos])
            
            # 각 관절 데이터 플로팅
            if joint_name in target_df.columns:
                ax.plot(t_target, target_df[joint_name].values, 'b-', linewidth=1.5, label='Target')
            
            if joint_name in user_df.columns:
                ax.plot(t_user, user_df[joint_name].values, 'r-', linewidth=1.5, label='User')
            
            if joint_name in generated_df.columns:
                ax.plot(t_gen, generated_df[joint_name].values, 'g-', linewidth=1.5, label='Generated')
            
            # 그래프 설정
            ax.set_xlabel('timestamp')
            ax.set_ylabel('deg')
            ax.set_title(joint_name)
            ax.grid(True)
            ax.legend()
        
        # 전체 제목 추가
        plt.suptitle(f'Trajectory type: {trajectory_type}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  

        # 그래프 표시
        if show:
            plt.show()
        else:
            plt.close()

    # def visualize_trajectories_animation(self, target_df, user_df, generated_df, trajectory_type, save_path=None, show=True):
    #     """궤적을 GIF로 저장 (타겟, 사용자, 생성된 궤적)"""
    #     import matplotlib.animation as animation
        
    #     # 그림 초기화
    #     fig = plt.figure(figsize=(20, 10))
    #     gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2]) 
        
    #     # 3D 그래프 생성
    #     ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
        
    #     # 타겟 궤적의 엔드이펙터 좌표 (정적)
    #     target_degrees = target_df[['deg1', 'deg2', 'deg3', 'deg4']].values
    #     target_endpoints = np.array([calculate_end_effector_position(deg) for deg in target_degrees]) * 1000
    #     target_x, target_y, target_z = target_endpoints[:, 0], target_endpoints[:, 1], target_endpoints[:, 2]
        
    #     # 사용자 궤적의 엔드이펙터 좌표 (정적)
    #     user_degrees = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
    #     user_endpoints = np.array([calculate_end_effector_position(deg) for deg in user_degrees]) * 1000
    #     user_x, user_y, user_z = user_endpoints[:, 0], user_endpoints[:, 1], user_endpoints[:, 2]
        
    #     # 생성된 궤적의 엔드이펙터 좌표 (애니메이션)
    #     gen_x = generated_df['x_end'].values 
    #     gen_y = generated_df['y_end'].values 
    #     gen_z = generated_df['z_end'].values 
        
    #     # 모든 x, y, z 좌표의 최소, 최대값을 구하여 동일한 축 범위 설정
    #     all_x = np.concatenate([target_x, user_x, gen_x])
    #     all_y = np.concatenate([target_y, user_y, gen_y])
    #     all_z = np.concatenate([target_z, user_z, gen_z])
        
    #     x_min, x_max = np.min(all_x), np.max(all_x)
    #     y_min, y_max = np.min(all_y), np.max(all_y)
    #     z_min, z_max = np.min(all_z), np.max(all_z)
        
    #     # 축 범위 설정 (약간의 여유 공간 추가)
    #     padding = 0.1
    #     x_range = x_max - x_min
    #     y_range = y_max - y_min
    #     z_range = z_max - z_min
        
    #     ax_3d.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    #     ax_3d.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    #     ax_3d.set_zlim(z_min - padding * z_range, z_max + padding * z_range)
        
    #     # 정적 궤적 그리기 (타겟, 사용자)
    #     ax_3d.plot(target_x, target_y, target_z, 'b-', linewidth=2, alpha=0.7, label='Target Trajectory')
    #     ax_3d.plot(user_x, user_y, user_z, 'r-', linewidth=2, alpha=0.7, label='User Trajectory')
        
    #     # 3D 그래프 설정
    #     ax_3d.set_xlabel('X')
    #     ax_3d.set_ylabel('Y')
    #     ax_3d.set_zlabel('Z')
    #     ax_3d.set_title('End-effector Trajectory Animation')
        
    #     # 오른쪽 영역을 2x2 서브플롯으로 분할
    #     gs_right = gs[0, 1].subgridspec(2, 2)
        
    #     # 애니메이션 프레임 수 설정
    #     num_frames = 100  # GIF 크기와 생성 시간을 고려하여 제한
        
    #     # 정적 라인 초기화
    #     joint_axes = []
    #     joint_target_lines = []
    #     joint_user_lines = []
    #     joint_gen_lines = []
    #     joint_gen_points = []
        
    #     # 시간 정규화 (0에서 1까지)
    #     t_target = np.linspace(0, 1, len(target_df))
    #     t_user = np.linspace(0, 1, len(user_df))
    #     t_gen = np.linspace(0, 1, len(generated_df))
        
    #     # 생성된 궤적을 위한 애니메이션 라인 객체
    #     gen_line, = ax_3d.plot([], [], [], 'g-', linewidth=2.5, label='Generated Trajectory')
    #     gen_point, = ax_3d.plot([], [], [], 'go', markersize=8)
        
    #     ax_3d.legend()
        
    #     # 관절 각도 그래프를 위한 서브플롯 초기화
    #     joint_names = ['deg1', 'deg2', 'deg3', 'deg4']
    #     positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  # 2x2 그리드 위치
        
    #     for i, (joint_name, pos) in enumerate(zip(joint_names, positions)):
    #         # 각 관절을 위한 서브플롯 생성
    #         ax = fig.add_subplot(gs_right[pos])
    #         joint_axes.append(ax)
            
    #         # 시간 범위 설정
    #         ax.set_xlim(0, 1)
            
    #         # 각도 범위 결정 (모든 궤적의 최대/최소값 고려)
    #         all_angles = []
    #         if joint_name in target_df.columns:
    #             all_angles.extend(target_df[joint_name].values)
    #         if joint_name in user_df.columns:
    #             all_angles.extend(user_df[joint_name].values)
    #         if joint_name in generated_df.columns:
    #             all_angles.extend(generated_df[joint_name].values)
                
    #         if all_angles:
    #             angle_min, angle_max = np.min(all_angles), np.max(all_angles)
    #             angle_range = angle_max - angle_min
    #             ax.set_ylim(angle_min - 0.1 * angle_range, angle_max + 0.1 * angle_range)
            
    #         # 정적 궤적 그리기 (타겟, 사용자)
    #         if joint_name in target_df.columns:
    #             target_line, = ax.plot(t_target, target_df[joint_name].values, 'b-', 
    #                                 linewidth=1.5, alpha=0.7, label='Target')
    #             joint_target_lines.append(target_line)
            
    #         if joint_name in user_df.columns:
    #             user_line, = ax.plot(t_user, user_df[joint_name].values, 'r-', 
    #                                 linewidth=1.5, alpha=0.7, label='User')
    #             joint_user_lines.append(user_line)
            
    #         # 생성된 궤적을 위한 애니메이션 라인 객체
    #         gen_joint_line, = ax.plot([], [], 'g-', linewidth=2, label='Generated')
    #         gen_joint_point, = ax.plot([], [], 'go', markersize=6)
            
    #         joint_gen_lines.append(gen_joint_line)
    #         joint_gen_points.append(gen_joint_point)
            
    #         # 그래프 설정
    #         ax.set_xlabel('Normalized Time')
    #         ax.set_ylabel('Angle (deg)')
    #         ax.set_title(joint_name)
    #         ax.grid(True)
    #         ax.legend()
        
    #     # 전체 제목 추가
    #     plt.suptitle(f'Trajectory Type: {trajectory_type}', fontsize=16)
    #     plt.tight_layout(rect=[0, 0, 1, 0.96])
        
    #     # 애니메이션 초기화 함수
    #     def init():
    #         gen_line.set_data([], [])
    #         gen_line.set_3d_properties([])
    #         gen_point.set_data([], [])
    #         gen_point.set_3d_properties([])
            
    #         for i in range(4):
    #             joint_gen_lines[i].set_data([], [])
    #             joint_gen_points[i].set_data([], [])
                
    #         return (gen_line, gen_point, *joint_gen_lines, *joint_gen_points)
        
    #     # 애니메이션 업데이트 함수
    #     def update(frame):
    #         # 프레임 비율 계산 (0~1 범위)
    #         t = frame / (num_frames - 1)
            
    #         # 생성된 궤적 업데이트
    #         idx_gen = min(int(t * len(generated_df)), len(generated_df) - 1)
    #         gen_line.set_data(gen_x[:idx_gen+1], gen_y[:idx_gen+1])
    #         gen_line.set_3d_properties(gen_z[:idx_gen+1])
            
    #         gen_point.set_data([gen_x[idx_gen]], [gen_y[idx_gen]])
    #         gen_point.set_3d_properties([gen_z[idx_gen]])
            
    #         # 관절 각도 그래프 업데이트 (생성된 궤적만)
    #         for i, joint_name in enumerate(joint_names):            
    #             # 생성된 궤적 관절 각도
    #             if joint_name in generated_df.columns:
    #                 joint_gen_lines[i].set_data(t_gen[:idx_gen+1], generated_df[joint_name].values[:idx_gen+1])
    #                 joint_gen_points[i].set_data([t_gen[idx_gen]], [generated_df[joint_name].values[idx_gen]])
            
    #         return (gen_line, gen_point, *joint_gen_lines, *joint_gen_points)
        
    #     # 애니메이션 생성
    #     frames = num_frames
    #     interval = 50  # 50ms 간격 (20 FPS)
    #     ani = animation.FuncAnimation(fig, update, frames=frames, 
    #                                 init_func=init, interval=interval, 
    #                                 blit=True)
        
    #     # 애니메이션을 GIF로 저장 (save_path가 제공된 경우)
    #     if save_path:
    #         try:
    #             # 저장 디렉토리가 없으면 생성
    #             save_dir = os.path.dirname(save_path)
    #             if save_dir and not os.path.exists(save_dir):
    #                 os.makedirs(save_dir)
                
    #             # 파일 확장자가 .gif가 아니면 변경
    #             if not save_path.lower().endswith('.gif'):
    #                 save_path = save_path.rsplit('.', 1)[0] + '.gif'
                
    #             # Pillow 라이터를 사용하여 GIF로 저장
    #             print(f"애니메이션을 GIF로 저장 중... (시간이 다소 소요될 수 있습니다)")
    #             ani.save(save_path, writer='pillow', fps=10, dpi=80)
    #             print(f"애니메이션이 GIF로 저장되었습니다: {save_path}")
    #         except Exception as e:
    #             print(f"GIF 저장 중 오류가 발생했습니다: {str(e)}")
    #             print("해상도나 프레임 수를 줄여서 다시 시도해보세요.")
        
    #     # 그래프 표시
    #     if show:
    #         plt.show()
    #     else:
    #         plt.close()
        
    #     return ani

def main():
    print("\n======== Trajectory Generation Mode ========")
    
    # 디렉토리 및 모델 경로 설정
    base_dir = "data"
    model_path = "best_trajectory_transformer_250.pth"
    output_dir = "generation_trajectory"
    
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
        print(f"Error: Failed to initialize analyzer: {str(e)}")
        return False
    
    # 생성기 초기화
    generator = EndeffectorGenerator(analyzer, model_path=model_path)

    # 사용자 궤적 파일 선택
    non_golden_dir = os.path.join(base_dir, "non_golden_sample")
    non_golden_files = [f for f in os.listdir(non_golden_dir) if f.endswith('.txt')]
    
    if not non_golden_files:
        print("User trajectory file not found.")
        return False
    
    # 랜덤하게 하나의 사용자 궤적 선택
    selected_file = random.choice(non_golden_files)
    
    # 사용자 궤적 로드 및 분류
    user_path = os.path.join(non_golden_dir, selected_file)
    user_trajectory, trajectory_type = analyzer.load_user_trajectory(user_path)
    print(f"\nSelected User Trajectory : {trajectory_type}")

    # 해당 타입의 타겟 궤적 찾기
    golden_dir = os.path.join(base_dir, "golden_sample")
    golden_files = [f for f in os.listdir(golden_dir) if trajectory_type in f and f.endswith('.txt')]
    
    if not golden_files:
        print(f"{trajectory_type} could not find target trajectory of type .")
        return False
    
    # 타겟 궤적 로드
    target_file = golden_files[0]
    print(f"Matched Target Trajectory: {target_file}")
    target_path = os.path.join(golden_dir, target_file)
    target_trajectory, _ = analyzer.load_user_trajectory(target_path)

    # 트랜스포머 모델을 사용하여 궤적 생성
    print("\nGenerating Trajectories With Transformer Models...")
    generated_df, results = generator.generate_trajectory(
        target_df=target_trajectory, 
        user_df=user_trajectory, 
        trajectory_type=trajectory_type
    )
    
    if generated_df is None:
        print("Trajectory creation failed")
        return False
    
    print(f"궤적 생성 완료: {len(generated_df)} 포인트")

    # 생성된 궤적 csv 저장
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{trajectory_type}_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    saved_path = generator.save_trajectory_to_csv(
        generated_df=generated_df,
        file_path=csv_path,
        trajectory_type=trajectory_type
    )
    
    if saved_path:
        print(f"궤적이 CSV 파일로 저장되었습니다: {saved_path}")

    # 애니메이션 파일 저장 경로 설정
    animation_filename = f"{trajectory_type}_{timestamp}.mp4"
    animation_path = os.path.join(output_dir, animation_filename)
    
    # 생성된 궤적 애니메이션 시각화
    # print("\n궤적 애니메이션 생성 중...")
    # try:
    #     _ = generator.visualize_trajectories_animation(
    #         target_df=target_trajectory,
    #         user_df=user_trajectory, 
    #         generated_df=generated_df,
    #         trajectory_type=trajectory_type,
    #         save_path=animation_path,
    #         show=True
    #     )
    #     print(f"애니메이션이 저장되었습니다: {animation_path}")
    # except Exception as e:
    #     print(f"애니메이션 생성 중 오류가 발생했습니다: {str(e)}")
        
    # 생성된 궤적 시각화
    print("\n궤적 시각화 중...")
    generator.visualize_trajectories(
        target_df=target_trajectory,
        user_df=user_trajectory, 
        generated_df=generated_df,
        trajectory_type=trajectory_type,
        save_path=None,
        show=True
    )
    
    print("\n처리 완료!")
    return True

if __name__ == "__main__":
    main()