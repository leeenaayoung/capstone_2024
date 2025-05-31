import os
import random
import datetime
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
        """ 트랜스포머 기반 엔드이펙터 궤적 생성 """
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
        """ 트랜스포머 모델 로드"""
        try:
            # 모델 초기화
            transformer = TrajectoryTransformer(
                            input_dim=7,
                            d_model=64,    
                            nhead=2,         
                            num_layers=2    
                        )
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                transformer.load_state_dict(checkpoint['model_state_dict'])
                # print(f"모델 가중치 로드 완료: epoch {checkpoint.get('epoch', 'unknown')}, loss {checkpoint.get('loss', 'unknown')}")
            else:
                transformer.load_state_dict(checkpoint)
                print(f"Model weights have been loaded")
            
            # 평가 모드 설정
            transformer.to(self.device)
            transformer.eval()
            
            # print(f"트랜스포머 모델 로드 성공: {model_path}")
            return transformer
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

    def smooth_data(self, data, smoothing_factor=1.0, degree=3):
        """궤적 스플라인 스무딩"""
        # 원본 데이터 복사
        smoothed_df = data.copy()

        # 관절 각도 추출
        degrees = data[['deg1', 'deg2', 'deg3', 'deg4']].values

        # 엔드이펙터 위치 계산
        endpoints = np.array([calculate_end_effector_position(deg) for deg in degrees]) * 1000

        # 엔드이펙터 위치를 x, y, z로 분리
        x = endpoints[:, 0]
        y = endpoints[:, 1]
        z = endpoints[:, 2]
        t = np.arange(len(data))

        # 스플라인 스무딩 적용
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

        # 리샘플링
        def resample(data, t_old, t_new):
            resampled = np.zeros((len(t_new), data.shape[1]))
            for i in range(data.shape[1]):
                spline = CubicSpline(t_old, data[:, i], bc_type='clamped')
                resampled[:, i] = spline(t_new)
            return resampled

        normalized_target = resample(aligned_target, num_points)
        normalized_user = resample(aligned_user, num_points)

        return normalized_target, normalized_user

    def generate_with_fixed_weight(self, user_tensor, target_tensor, fixed_weight):
        """ 지정된 가중치를 사용하여 궤적 생성 """
        # 배치 크기,시퀀스 길이, 입력 차원 불러오기
        batch_size, seq_len, input_dim = user_tensor.size()
        
        # 사용자와 타겟 궤적을 각각 임베딩 처리
        user_emb = self.transformer.positional_encoding(
            self.transformer.user_embedding(user_tensor)
        )
        target_emb = self.transformer.positional_encoding(
            self.transformer.target_embedding(target_tensor)
        )
        
        # 평가 결과를 기반으로 한 가중치 조정
        weight_tensor = torch.tensor([fixed_weight], device=user_tensor.device).float()
        weight_tensor = weight_tensor.unsqueeze(0).unsqueeze(-1) 

        memory = weight_tensor * user_emb + (1 - weight_tensor) * target_emb

        if hasattr(self.transformer, 'temporal_encoder'):
            memory = memory + self.transformer.temporal_encoder(memory)
        
        # 생성
        generated = torch.zeros(batch_size, seq_len, input_dim).to(user_tensor.device)
        
        generated[:, 0, :] = (
            weight_tensor.squeeze() * user_tensor[:, 0, :] + 
            (1 - weight_tensor.squeeze()) * target_tensor[:, 0, :]
        )

        for t in range(1, seq_len):
            current_input = generated[:, :t, :]

            tgt_emb = self.transformer.output_embedding(current_input)
            tgt_emb = self.transformer.positional_encoding(tgt_emb)

            tgt_mask = self.transformer._generate_square_subsequent_mask(t).to(user_tensor.device)

            decoder_output = self.transformer.decoder(tgt_emb, memory[:, :t, :], tgt_mask=tgt_mask)
            next_step = self.transformer.output_layer(decoder_output[:, -1:, :])
            
            # 예측된 값 저장
            generated[:, t, :] = next_step.squeeze(1)
        
        return generated
    
    def prepare_trajectory_input(self, target_df, user_df, max_seq_length=100):
        """트랜스포머 모델 입력을 위한 궤적 데이터 준비 (위치/각도 분리 정규화)"""
        def process_trajectory_inference(df):
            degrees = df[['deg1', 'deg2', 'deg3', 'deg4']].values
            endpoints = np.array([calculate_end_effector_position(deg) for deg in degrees]) * 1000
            combined = np.concatenate([endpoints, degrees], axis=1)
            
            t_orig = np.linspace(0, 1, len(combined))
            t_new = np.linspace(0, 1, max_seq_length)
            
            resampled = np.zeros((max_seq_length, combined.shape[1]))
            for i in range(combined.shape[1]):
                spline = CubicSpline(t_orig, combined[:, i])
                resampled[:, i] = spline(t_new)
            
            mean = np.mean(resampled, axis=0)
            std = np.std(resampled, axis=0) + 1e-8
            normalized = (resampled - mean) / std
            
            return normalized, mean, std, resampled 
        
        target_normalized, target_mean, target_std, target_raw = process_trajectory_inference(target_df)
        user_normalized, user_mean, user_std, user_raw = process_trajectory_inference(user_df)
        
        alpha = 0.5
        interpolated_gt = alpha * user_normalized + (1 - alpha) * target_normalized
        
        # 역정규화를 위한 모든 정보를 저장
        normalization_params = {
            'user_mean': user_mean,
            'user_std': user_std,
            'target_mean': target_mean,
            'target_std': target_std,
            'user_raw': user_raw,
            'target_raw': target_raw,
            'max_seq_length': max_seq_length
        }
        
        target_tensor = torch.FloatTensor(target_normalized).unsqueeze(0).to(self.device)
        user_tensor = torch.FloatTensor(user_normalized).unsqueeze(0).to(self.device)
        interpolated_tensor = torch.FloatTensor(interpolated_gt).unsqueeze(0).to(self.device)
        
        return user_tensor, target_tensor, interpolated_tensor, normalization_params

    def denormalize_with_interpolation(self, generated_np, normalization_params, weight):
        """가중치를 고려한 적절한 역정규화"""
        # 원본 데이터에서 직접 보간
        user_raw = normalization_params['user_raw']
        target_raw = normalization_params['target_raw']
        interpolated_raw = weight * user_raw + (1 - weight) * target_raw
        
        # 위치와 각도 분리
        generated_pos = interpolated_raw[:, :3]  
        generated_deg = interpolated_raw[:, 3:]  
        
        return generated_pos, generated_deg

    def generate_trajectory(self, target_df, user_df, trajectory_type, weight=None):
        """트랜스포머 모델을 사용하여 새로운 궤적 생성"""

        if self.transformer is None:
            print("Transformer model not loaded.")
            return None, None

        try:
            # 데이터 전처리
            user_tensor, target_tensor, interpolated_tensor, norm_params = self.prepare_trajectory_input(
                target_df, user_df)

            # 모델을 통한 궤적 생성
            with torch.no_grad(): 
                if weight is not None:
                    # 평가 결과에 따른 특정 가중치가 제공
                    print(f"Attempting to generate trajectory with specified weights {weight:.2f}...")
                    generated = self.generate_with_fixed_weight(
                        user_tensor, target_tensor, weight
                    )
                    
                    # print(f"지정된 가중치 {weight:.2f}가 정확히 적용됨")
                    predicted_weight = torch.tensor([weight])
                    
                else:
                    print("Generating trajectories using the model's adaptive weights...")
                    result = self.transformer(user_tensor, target_tensor, interpolated_gt=None)
                    
                    # 모델 출력 형태에 따른 처리
                    if isinstance(result, tuple):
                        generated, predicted_weight = result
                    else:
                        generated = result
                        predicted_weight = None
                    # print("모델의 적응적 가중치 사용됨")

            generated_np = generated.squeeze(0).cpu().numpy()

            # 역정규화 과정
            generated_pos, generated_deg = self.denormalize_with_interpolation(
                generated_np, norm_params, weight if weight is not None else 0.5
            )

            # 물리적 제약 조건 적용
            for joint_idx, (min_angle, max_angle) in self.joint_limits.items():
                generated_deg[:, joint_idx] = np.clip(
                    generated_deg[:, joint_idx], min_angle, max_angle
                )

            # 결과 반환
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
                'transformer_used': True,
                'applied_weight': weight if weight is not None else 'adaptive'
            }

            # print("생성된 궤적에 스무딩 적용 중...")
            generated_df = self.smooth_data(generated_df, smoothing_factor=0.5, degree=3)
            return generated_df, results

        except Exception as e:
            print(f"Error occurred while generating trajectory: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
        
    def save_trajectory_to_csv(self, generated_df, file_path=None, trajectory_type="unknown"):
        """ 생성된 궤적 데이터프레임 저장 """
        required_columns = ['x_end', 'y_end', 'z_end', 'deg1', 'deg2', 'deg3', 'deg4']
        missing_columns = [col for col in required_columns if col not in generated_df.columns]
        
        if missing_columns:
            print(f"The following column is not in the dataframe: {missing_columns}")

            for col in missing_columns:
                generated_df[col] = 0.0
                print(f"- '{col}' added a column filled with 0.")
        
        # 열 순서 정렬
        final_df = generated_df[required_columns].copy()

        if 'timestamps' in generated_df.columns:
            final_df['timestamps'] = generated_df['timestamps']
        
        if file_path is None:

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"generated_trajectory_{trajectory_type}_{timestamp}.csv"
        
        # CSV 파일로 저장
        try:
            final_df.to_csv(file_path, index=False)
            print(f"Trajectory data was successfully saved: {file_path}")
            return file_path
        except Exception as e:
            print(f"An error occurred while saving the file: {str(e)}")
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
    print("\n======== Rehabilitation Trajectory Generation ========")

    base_dir = "data"
    model_path = "best_trajectory_transformer.pth"
    output_dir = "generation_trajectory"

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
    target_trajectory, target_file = analyzer.load_target_trajectory(trajectory_type)

    evaluation_result = evaluator.evaluate_trajectory(user_trajectory, trajectory_type)
    golden_dict = load_golden_evaluation_results(trajectory_type, base_dir)
    final_score = calculate_score_with_golden(evaluation_result, golden_dict)
    grade = convert_score_to_rank(final_score)
    weight = generator.grade_to_weight.get(grade, 0.8)

    print(f"[Final Score] => {final_score:.2f} / 100")
    print(f"[Final Grade] => Grade {grade}")
    print(f"[Applied Interpolation Weight] => {weight:.1f}")

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

    # 생성된 궤적 csv 저장
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{trajectory_type}_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    saved_path = generator.save_trajectory_to_csv(
        generated_df=generated_df,
        file_path=csv_path,
        trajectory_type=trajectory_type
    )
    
    # if saved_path:
    #     print(f"궤적이 CSV 파일로 저장되었습니다: {saved_path}")


    print("\nVisualizing trajectories...")
    generator.visualize_trajectories(
        target_df=target_trajectory,
        user_df=user_trajectory, 
        generated_df=generated_df,
        trajectory_type=trajectory_type,
        save_path="generated_trajectory.png",
        show=True
    )

    print("\nProcessing completed!")
    return True

if __name__ == "__main__":
    main()