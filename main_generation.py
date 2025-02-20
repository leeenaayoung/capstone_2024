import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from generation import TrajectoryGenerator
from analyzer import TrajectoryAnalyzer
from generation_model import JointTrajectoryTransformer
from utils import calculate_end_effector_position

class ImprovedTrajectoryGenerator(TrajectoryGenerator):
    def __init__(self, analyzer, generation_model_path='trajectory_generation_model.pth', device=None):
        # 부모 클래스 초기화로 normalize_time 등의 메서드 상속
        super().__init__(analyzer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.correlation_model = self._initialize_model(generation_model_path)

    def _initialize_model(self, model_path):
        """상관관계 모델 초기화 및 로드"""
        model = JointTrajectoryTransformer().to(self.device)
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print("Correlation model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")
        return model

    def interpolate_trajectory(self, target_df, user_df, trajectory_type):
        """개선된 보간 함수: DTW 정규화와 상관관계 모델 통합"""
        # 1단계: 기존 방식으로 초기 보간 수행
        initial_df = super().interpolate_trajectory(target_df, user_df, trajectory_type)
        
        # 2단계: 데이터 준비 및 DTW 정규화
        # 각도 데이터 추출
        target_angles = target_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_angles = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        initial_angles = initial_df[['deg1', 'deg2', 'deg3', 'deg4']].values

        # DTW 정규화를 위해 속도 데이터를 포함한 전체 데이터 준비
        target_full = np.column_stack([target_angles, np.zeros_like(target_angles)])  # 속도는 0으로 패딩
        initial_full = np.column_stack([initial_angles, np.zeros_like(initial_angles)])
        
        # normalize_time 메서드를 사용하여 시간 정규화
        aligned_target, aligned_initial = self.normalize_time(target_full, initial_full)
        
        # 정규화된 각도 데이터 추출
        aligned_target_angles = aligned_target[:, :4]
        aligned_initial_angles = aligned_initial[:, :4]

        # 3단계: 상관관계 모델 적용 및 조정
        with torch.no_grad():
            # 정규화된 데이터를 모델에 입력
            input_tensor = torch.FloatTensor(aligned_initial_angles).unsqueeze(0).to(self.device)
            model_output = self.correlation_model(input_tensor).squeeze(0).cpu().numpy()
            
            # 원본과의 거리에 기반한 가중치 계산
            target_dist = np.mean(np.abs(aligned_target_angles - aligned_initial_angles), axis=1)
            total_dist = target_dist / target_dist.max()  # 정규화
            
            # 시그모이드 함수를 사용한 부드러운 가중치 계산
            weights = 1 / (1 + np.exp(-5 * (total_dist - 0.5)))
            weights = weights.reshape(-1, 1)
            
            # 최종 각도 계산 - 부드러운 보간
            final_angles = (1 - weights) * aligned_initial_angles + weights * model_output

            # Joint limit 적용
            for joint in range(4):
                min_val, max_val = self.joint_limits[joint]
                final_angles[:, joint] = np.clip(final_angles[:, joint], min_val, max_val)

        # 4단계: End-effector 위치 계산 및 결과 생성
        endeffector_points = np.array([calculate_end_effector_position(deg) for deg in final_angles])
        endeffector_points = endeffector_points * 1000

        # 결과 데이터프레임 생성
        generated_df = pd.DataFrame(
            np.column_stack([endeffector_points, final_angles]),
            columns=['x_end', 'y_end', 'z_end', 'deg1', 'deg2', 'deg3', 'deg4']
        )

        return generated_df

def main():
    base_dir = os.path.join(os.getcwd(), "data")
    
    try:
        # 분석기 객체 초기화
        analyzer = TrajectoryAnalyzer(
            classification_model="best_classification_model.pth",
            base_dir=base_dir
        )
        
        # 개선된 생성기 초기화
        final_generator = ImprovedTrajectoryGenerator(
            analyzer=analyzer,
            generation_model_path='trajectory_generation_model.pth'
        )

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
        
        # 보간 및 정제가 통합된 궤적 생성
        print("\nGenerating trajectory with correlation model...")
        generated_df = final_generator.interpolate_trajectory(
            target_df=target_trajectory,
            user_df=user_trajectory,
            trajectory_type=trajectory_type
        )

        # 시각화 및 저장
        print("\nVisualizing and saving trajectories...")
        final_generator.visualize_trajectories(
            target_df=target_trajectory,
            user_df=user_trajectory,
            generated_df=generated_df,
            trajectory_type=trajectory_type
        )
        
        print("\nProcessing completed!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()