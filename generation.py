import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import MinMaxScaler
from fastdtw import fastdtw
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
import os
import random
from utils import *
from model import ClassificationDataset

class TrajectoryAnalyzer:
    def __init__(self, classification_model: str = "best_classification_model.pth", base_dir="data"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        c_base_path = "data/all_data"
        c_dataset = ClassificationDataset(c_base_path)
        print("Available labels", c_dataset.unique_labels)
        self.trajectory_types = {i: label for i, label in enumerate(c_dataset.unique_labels)}
        print("\nGenerated trajectory_types:", self.trajectory_types)
        self.classifier = self.load_classifier(classification_model)
        self.base_dir = base_dir

    def load_classifier(self, model_path : str):
        """ 분류 모델 로드 """
        try:
            from model import TransformerModel
            model = TransformerModel(
                input_dim=21,      
                d_model=32,       
                nhead=2,           
                num_layers=3,      
                num_classes=len(self.trajectory_types)
            ).to(self.device)

            # 저장된 state_dict 로드
            state_dict = torch.load(model_path, map_location=self.device)
            print("state_dict keys:", state_dict.keys())
            
            # 가중치를 모델에 적용
            model.load_state_dict(state_dict)
            
            # 평가 모드로 설정
            model.eval()
            
            print(f"Successfully loaded classification model: {model_path}")
            return model
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def load_user_trajectory(self, file_path: str = "data/non_golden_sample"):
        """ 사용자 궤적 로드 """
        try:
            df = pd.read_csv(file_path, delimiter=',')

            processed_df = preprocess_trajectory_data(df)

            tensor_data = torch.FloatTensor(processed_df.values).unsqueeze(0)
            tensor_data = tensor_data.to(self.device)
            
            with torch.no_grad(): 
                predictions = self.classifier(tensor_data)
                predicted_class = torch.argmax(predictions, dim=1).item()

                print(f"Predicted Class Index: {predicted_class}")
                # print(f"예측 확률 분포: {torch.softmax(predictions, dim=1)}")
                
                # trajectory_types에서 해당 클래스 찾기
                if predicted_class in self.trajectory_types:
                    predicted_type = self.trajectory_types[predicted_class]
                else:
                    raise ValueError(f"Predicted Class Index {predicted_class}is not in the trajectory_types")
            
            print(f"Classification Result : ")
            print(f"{predicted_type}")
            
            return processed_df, predicted_type
            
        except Exception as e:
            print(f"Trajectory file {file_path} error during processing: {str(e)}")
            raise

    def load_target_trajectory(self, trajectory_type: str):
        """ user_trajectory와 같은 타입의 target_trajectory 로드"""
        try:
            # 해당 타입으로 시작하는 모든 파일을 찾습니다
            matching_files = [f for f in os.listdir(self.golden_dir) 
                            if f.startswith(trajectory_type) and f.endswith('.txt')]
            
            if not matching_files:
                raise ValueError(f"From the golden_sample directory {trajectory_type} can't find the trajectory of the type")
            
            # 매칭되는 파일들 중 하나를 무작위로 선택합니다
            selected_file = random.choice(matching_files)
            file_path = os.path.join(self.golden_dir, selected_file)
            
            # 선택된 파일을 로드하고 전처리합니다
            df = pd.read_csv(file_path, delimiter=',')
            processed_df = preprocess_trajectory_data(df)
            
            print(f"\nLoaded Target Trajectory:")
            print(f"selected file: {selected_file}")
            
            return processed_df, selected_file
            
        except Exception as e:
            print(f"Error loading target trajectory: {str(e)}")
            raise

# 궤적 생성, 변환, 시각화
class TrajectoryGenerator:
    def __init__(self):
        pass

    @staticmethod
    def smooth_data(data, sigma=10):
        """가우시안 필터를 사용한 라벨 스무딩"""
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(data, sigma=sigma)

    @staticmethod
    def normalize_time(trajectory, num_points=100):
        """시간에 대해 궤적을 정규화"""
        current_length = len(trajectory)
        old_time = np.linspace(0, 1, current_length)
        new_time = np.linspace(0, 1, num_points)

        interpolator_x = interp1d(old_time, trajectory[:, 0], kind='cubic')
        interpolator_y = interp1d(old_time, trajectory[:, 1], kind='cubic')
        interpolator_z = interp1d(old_time, trajectory[:, 2], kind='cubic')

        return np.column_stack((
            interpolator_x(new_time),
            interpolator_y(new_time),
            interpolator_z(new_time)
        ))

    def apply_dtw(self, target, subject, interpolation_weight=0.5):
        """ DTW를 적용하여 궤적을 정렬하고 보간 """

        # 시간 정규화
        target_norm = self.normalize_time(target)
        subject_norm = self.normalize_time(subject)

        # 스무딩 적용
        target_smoothed = np.zeros_like(target_norm)
        subject_smoothed = np.zeros_like(subject_norm)
        for i in range(3):
            target_smoothed[:, i] = self.smooth_data(target_norm[:, i])
            subject_smoothed[:, i] = self.smooth_data(subject_norm[:, i])

        # DTW 거리 및 경로 계산
        distance, path = fastdtw(target_smoothed, subject_smoothed, dist=euclidean)
        path = np.array(path)

        # 매칭된 포인트들 추출
        target_matched = target_smoothed[path[:, 0]]
        subject_matched = subject_smoothed[path[:, 1]]

        # 보간된 궤적 생성
        interpolated = (target_matched * interpolation_weight + 
                       subject_matched * (1 - interpolation_weight))

        # 결과 궤적을 원본 길이로 리샘플링
        return self.normalize_time(interpolated, num_points=len(target))

    def compare_trajectories(self, target_df, user_df, save_animation=False):
        """ 타겟과 사용자 궤적을 비교하고 정렬된 궤적을 생성하여 시각화 """

        # 원본 궤적 포인트 추출
        target_points = target_df[['x_end', 'y_end', 'z_end']].values
        user_points = user_df[['x_end', 'y_end', 'z_end']].values
        
        # DTW를 사용하여 정렬된 궤적 생성
        aligned_trajectory = self.apply_dtw(target_points, user_points)
        
        # 세 궤적 모두 시각화
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 타겟 궤적
        ax.plot(target_points[:, 0], target_points[:, 1], target_points[:, 2],
                'b--', label='Target Trajectory')
        
        # 원본 사용자 궤적
        ax.plot(user_points[:, 0], user_points[:, 1], user_points[:, 2],
                'r--', label='Original Subject Trajectory')
        
        # 정렬된 궤적
        ax.plot(aligned_trajectory[:, 0], aligned_trajectory[:, 1], aligned_trajectory[:, 2],
                'g-', label='Aligned Subject Trajectory', linewidth=2)

        # 그래프 설정
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title('Trajectory Alignment using DTW')
        ax.view_init(10, 90)
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper right')
        ax.grid(True)

        # 모든 궤적을 포함하도록 축 범위 설정
        all_points = np.vstack([target_points, user_points, aligned_trajectory])
        margin = 10  # 여백 추가
        ax.set_xlim([min(all_points[:, 0]) - margin, max(all_points[:, 0]) + margin])
        ax.set_ylim([min(all_points[:, 1]) - margin, max(all_points[:, 1]) + margin])
        ax.set_zlim([min(all_points[:, 2]) - margin, max(all_points[:, 2]) + margin])

        plt.show()
        
        return aligned_trajectory
    
def main():
    # 기본 디렉토리 설정
    base_dir = os.path.join(os.getcwd(), "data")
    
    try:
        # TrajectoryAnalyzer와 TrajectoryGenerator 객체 초기화
        analyzer = TrajectoryAnalyzer(
            classification_model="best_classification_model.pth",
            base_dir=base_dir
        )
        generator = TrajectoryGenerator()
        
        # 1. user trajectory 불러오기
        print("\nLoading and classifying user trajectories...")
        non_golden_dir = os.path.join(base_dir, "non_golden_sample")
        non_golden_files = [f for f in os.listdir(non_golden_dir) 
                          if f.endswith('.txt')]
        
        if not non_golden_files:
            raise ValueError("The trajectory file is missing in the non_golden_sample directory.")
        
        # 사용자 궤적 선택(가정)
        selected_file = random.choice(non_golden_files)
        print(f"Selected user trajectory: {selected_file}")
        
        # 선택된 사용자 궤적 로드 및 분류
        file_path = os.path.join(non_golden_dir, selected_file)
        user_trajectory, trajectory_type = analyzer.load_user_trajectory(file_path)
        
        # 타겟 궤적 로드
        print("\nSearching for matching target trajectories...")
        golden_dir = os.path.join(base_dir, "golden_sample")
        golden_files = [f for f in os.listdir(golden_dir) 
                       if f.startswith(trajectory_type) and f.endswith('.txt')]
        
        if not golden_files:
            raise ValueError(f"'{trajectory_type}' target trajectory of type not found.")

        selected_golden = random.choice(golden_files)
        golden_path = os.path.join(golden_dir, selected_golden)
        target_trajectory = pd.read_csv(golden_path, delimiter=',')
        target_trajectory = preprocess_trajectory_data(target_trajectory)
        print(f"Selected Target Trajectory: {selected_golden}")
        
        # 궤적 비교 및 시각화
        print("\nCompare and visualize trajectories")
        aligned_trajectory = generator.compare_trajectories(
            target_df=target_trajectory,
            user_df=user_trajectory
        )
        
        print("\nProcessing completed!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Please check the data directory structure and model path")

if __name__ == "__main__":
    main()