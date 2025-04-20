import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.interpolate import UnivariateSpline, CubicSpline
from scipy.signal import savgol_filter
from utils import calculate_end_effector_position
from analyzer import TrajectoryAnalyzer
from endeffector_model import TrajectoryTransformer
from evaluate import *

class EndeffectorGenerator:
    def __init__(self, analyzer, model_path="best_trajectory_transformer.pth"):
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
                            d_model=128,     # 훈련 시 설정한 d_model
                            nhead=4,         # 훈련 시 설정한 nhead
                            num_layers=4     # 훈련 시 설정한 num_layers
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
            
    def smooth_data(self, data, window_length=11, polyorder=3):
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

    # def smooth_data(self, data, smoothing_factor=1.0, degree=3):
    #     """궤적 스플라인 스무딩"""
    #     # 원본 데이터 복사
    #     smoothed_df = data.copy()

    #     # 관절 각도 추출
    #     degrees = data[['deg1', 'deg2', 'deg3', 'deg4']].values

    #     # 엔드이펙터 위치 계산
    #     endpoints = np.array([calculate_end_effector_position(deg) for deg in degrees])

    #     # 엔드이펙터 위치를 x, y, z로 분리
    #     x = endpoints[:, 0]
    #     y = endpoints[:, 1]
    #     z = endpoints[:, 2]
    #     t = np.arange(len(data))

    #     # 스플라인 스무딩 적용
    #     s = smoothing_factor * len(data)
    #     spline_x = UnivariateSpline(t, x, k=degree, s=s)
    #     spline_y = UnivariateSpline(t, y, k=degree, s=s)
    #     spline_z = UnivariateSpline(t, z, k=degree, s=s)

    #     # 스무딩된 엔드이펙터 위치
    #     smoothed_endpoints = np.column_stack([spline_x(t), spline_y(t), spline_z(t)])

    #     # 결과를 DataFrame에 저장
    #     smoothed_df['x'] = smoothed_endpoints[:, 0]
    #     smoothed_df['y'] = smoothed_endpoints[:, 1]
    #     smoothed_df['z'] = smoothed_endpoints[:, 2]

    #     return smoothed_df
    
    def normalize_time(self, target_data, user_data, num_points=100):
        """시간 정규화 및 DTW 적용"""
        # 스무딩 적용
        target_smoothed = self.smooth_data(target_data, smoothing_factor=0.5, degree=3)
        user_smoothed = self.smooth_data(user_data, smoothing_factor=0.5, degree=3)
        
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

        # 균일한 길이로 리샘플링
        # def resample(points, n):
        #     t = np.linspace(0, 1, len(points))
        #     t_new = np.linspace(0, 1, n)
        #     x = UnivariateSpline(t, points[:, 0], s=0)(t_new)
        #     y = UnivariateSpline(t, points[:, 1], s=0)(t_new)
        #     z = UnivariateSpline(t, points[:, 2], s=0)(t_new)
        #     return np.column_stack([x, y, z])
        
        # 수정
        def resample(data, t_old, t_new):
            resampled = np.zeros((len(t_new), data.shape[1]))
            for i in range(data.shape[1]):
                # 경계 조건을 'clamped'로 줘서 자연스러운 시작과 끝 보장
                spline = CubicSpline(t_old, data[:, i], bc_type='clamped')
                resampled[:, i] = spline(t_new)
            return resampled

        normalized_target = resample(aligned_target, num_points)
        normalized_user = resample(aligned_user, num_points)

        return normalized_target, normalized_user
    
    def prepare_trajectory_input(self, target_df, user_df, max_seq_length=100):
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
    
    def generate_trajectory(self, target_df, user_df, trajectory_type, weight=None):
        """트랜스포머 모델을 사용하여 새로운 궤적 생성"""
        if self.transformer is None:
            print("트랜스포머 모델이 로드되지 않았습니다.")
            return None, None

        try:
            user_tensor, target_tensor, interpolated_tensor, norm_params = self.prepare_trajectory_input(
                target_df, user_df)

            with torch.no_grad():
                if weight is not None:
                    generated = self.transformer.generate_with_weight(user_tensor, target_tensor, weight=weight)
                else:
                    generated = self.transformer(user_tensor, target_tensor, interpolated_gt=None)

            generated_np = generated.squeeze(0).cpu().numpy()

            # print("Generated first 3 columns (assumed x, y, z):", generated_np[0, :3])
            # print("Generated degs (assumed deg1~4):", generated_np[0, 3:])

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
        
        # 관절 각도 그래프 (4개의 관절 각도를 2x2 서브플롯으로)
        joint_names = ['deg1', 'deg2', 'deg3', 'deg4']
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  # 2x2 그리드 위치
        
        for i, (joint_name, pos) in enumerate(zip(joint_names, positions)):
            # 각 관절을 위한 서브플롯 생성
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

def main():
    print("\n======== Trajectory Generation Mode ========")

    base_dir = "data"
    model_path = "best_trajectory_transformer.pth"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return False

    try:
        analyzer = TrajectoryAnalyzer(
            classification_model="best_classification_model.pth",
            base_dir=base_dir
        )
    except Exception as e:
        print(f"Error: Failed to initialize analyzer: {str(e)}")
        return False

    generator = EndeffectorGenerator(analyzer, model_path=model_path)
    evaluator = TrajectoryEvaluator()

    non_golden_dir = os.path.join(base_dir, "non_golden_sample")
    non_golden_files = [f for f in os.listdir(non_golden_dir) if f.endswith('.txt')]

    if not non_golden_files:
        print("User trajectory file not found.")
        return False

    selected_file = random.choice(non_golden_files)
    user_path = os.path.join(non_golden_dir, selected_file)
    user_trajectory, trajectory_type = analyzer.load_user_trajectory(user_path)
    print(f"\nSelected User Trajectory : {trajectory_type}")

    golden_dir = os.path.join(base_dir, "golden_sample")
    golden_files = [f for f in os.listdir(golden_dir) if trajectory_type in f and f.endswith('.txt')]

    if not golden_files:
        print(f"{trajectory_type} could not find target trajectory of type .")
        return False

    target_file = golden_files[0]
    print(f"Matched Target Trajectory: {target_file}")
    target_path = os.path.join(golden_dir, target_file)
    target_trajectory, _ = analyzer.load_user_trajectory(target_path)

    evaluation_result = evaluator.evaluate_trajectory(user_trajectory, trajectory_type)
    golden_dict = load_golden_evaluation_results(trajectory_type, base_dir)
    final_score = calculate_score_with_golden(evaluation_result, golden_dict)
    grade = convert_score_to_rank(final_score)
    weight = generator.grade_to_weight.get(grade, 0.8)

    print(f"[Final Score] => {final_score:.2f} / 100")
    print(f"[Final Grade] => {grade}등급")
    print(f"[Applied Interpolation Weight] => {weight:.2f}")

    print("\nGenerating Trajectories With Transformer Models...")
    generated_df, results = generator.generate_trajectory(
        target_df=target_trajectory, 
        user_df=user_trajectory, 
        trajectory_type=trajectory_type,
        weight=weight
    )

    if generated_df is None:
        print("Trajectory creation failed")
        return False

    print(f"궤적 생성 완료: {len(generated_df)} 포인트")

    print("\n궤적 시각화 중...")
    generator.visualize_trajectories(
        target_df=target_trajectory,
        user_df=user_trajectory, 
        generated_df=generated_df,
        trajectory_type=trajectory_type,
        save_path="generated_trajectory.png",
        show=True
    )

    print("\n처리 완료!")
    return True

if __name__ == "__main__":
    main()
