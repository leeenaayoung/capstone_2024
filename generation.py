import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
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
        self.c_dataset = ClassificationDataset(c_base_path)  
        print("Available labels", self.c_dataset.unique_labels)

        self.trajectory_types = {i: label for i, label in enumerate(self.c_dataset.unique_labels)}
        print("\nGenerated trajectory_types:", self.trajectory_types)

        self.classifier = self.load_classifier(classification_model)
        self.base_dir = base_dir
        self.golden_dir = os.path.join(self.base_dir, "golden_sample")

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
            
            print(f"Successfully loaded classification model : {model_path}")
            return model
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def load_user_trajectory(self, file_path: str = "data/non_golden_sample"):
        """ 사용자 궤적 로드 """
        try:
            df = pd.read_csv(file_path, delimiter=',')
            scaled_df, preprocessed_df = preprocess_trajectory_data(df, scaler=self.c_dataset.scaler, return_raw=True)

            # 분류 시 스케일링 적용된 데이터 사용
            tensor_data = torch.FloatTensor(scaled_df.values).unsqueeze(0)
            tensor_data = tensor_data.to(self.device)
            
            with torch.no_grad(): 
                predictions = self.classifier(tensor_data)
                predicted_class = torch.argmax(predictions, dim=1).item()
                
                # trajectory_types에서 해당 클래스 찾기
                if predicted_class in self.trajectory_types:
                    predicted_type = self.trajectory_types[predicted_class]
                else:
                    raise ValueError(f"Predicted Class Index {predicted_class}is not in the trajectory_types")
            
            print(f"Classification Result : {predicted_type}")

            return preprocessed_df, predicted_type
            
        except Exception as e:
            print(f"Trajectory file {file_path} error during processing: {str(e)}")
            raise

    def load_target_trajectory(self, trajectory_type: str):
        """ user_trajectory와 같은 타입의 target_trajectory 로드"""
        try:
            matching_files = [f for f in os.listdir(self.golden_dir) 
                            if f.startswith(trajectory_type) and f.endswith('.txt')]
            
            if not matching_files:
                raise ValueError(f"From the golden_sample directory {trajectory_type} can't find the trajectory of the type")
            
            # 매칭되는 파일들 중 하나를 무작위로 선택(타겟 궤적 하나로 수정)
            selected_file = random.choice(matching_files)
            file_path = os.path.join(self.golden_dir, selected_file)
            
            # 선택된 파일 로드 및 전처리
            df = pd.read_csv(file_path, delimiter=',')
            _, preprocessed_df = preprocess_trajectory_data(df, scaler=self.c_dataset.scaler, return_raw=True)

            
            return preprocessed_df, selected_file
            
        except Exception as e:
            print(f"Error loading target trajectory: {str(e)}")
            raise

# 궤적 생성, 변환, 시각화
class TrajectoryGenerator:
    def __init__(self):
        pass

    @staticmethod
    def smooth_data(data, sigma=10):
        """가우시안 필터를 사용한 데이터 스무딩"""
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(data, sigma=sigma)

    @staticmethod
    def normalize_time(trajectory, num_points=100):
        """시간에 대해 궤적을 정규화"""
        current_length = len(trajectory)
        old_time = np.linspace(0, 1, current_length)
        new_time = np.linspace(0, 1, num_points)

        interpolated_trajectory = np.zeros((num_points, trajectory.shape[1]))
        
        for i in range(trajectory.shape[1]):
            interpolator = interp1d(old_time, trajectory[:, i], kind='cubic')
            interpolated_trajectory[:, i] = interpolator(new_time)
        
        return interpolated_trajectory

    def apply_dtw(self, target, subject, interpolation_weight=0.5):
        """DTW를 적용하여 궤적을 정렬하고 보간"""
        # DTW 적용을 위한 데이터 준비 (스무딩만 적용)
        target_smoothed = np.zeros_like(target)
        subject_smoothed = np.zeros_like(subject)
        
        for i in range(target.shape[1]):
            target_smoothed[:, i] = self.smooth_data(target[:, i])
            subject_smoothed[:, i] = self.smooth_data(subject[:, i])

        # DTW 거리 및 경로 계산
        distance, path = fastdtw(target_smoothed, subject_smoothed, dist=euclidean)
        path = np.array(path)

        # 매칭된 포인트들 추출
        target_matched = target[path[:, 0]]  # 스무딩되지 않은 원본 데이터 사용
        subject_matched = subject[path[:, 1]]  # 스무딩되지 않은 원본 데이터 사용

        # 보간된 궤적 생성
        interpolated = (target_matched * interpolation_weight + 
                    subject_matched * (1 - interpolation_weight))

        # 목표 궤적의 길이에 맞춰 시간 정규화 적용
        return self.normalize_time(interpolated, num_points=len(target))

    def compare_trajectories(self, target_df, user_df, classification_result, generation_number=1):
        """타겟과 사용자 궤적을 비교하고 정렬된 궤적을 생성하여 시각화"""
        # 관절 각도 데이터로 DTW와 보간 수행
        target_degrees = target_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_degrees = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        aligned_degrees = self.apply_dtw(target_degrees, user_degrees)
        
        # 보간된 관절 각도를 이용하여 end-effector 위치 계산
        target_points = np.array([calculate_end_effector_position(deg) for deg in target_degrees])
        user_points = np.array([calculate_end_effector_position(deg) for deg in user_degrees])
        aligned_points = np.array([calculate_end_effector_position(deg) for deg in aligned_degrees])
        
        # 생성된 궤적의 데이터프레임 생성
        generated_df = pd.DataFrame(
            np.column_stack([aligned_points, aligned_degrees]),
            columns=['x_end', 'y_end', 'z_end', 'deg1', 'deg2', 'deg3', 'deg4']
        )
        # 시각화
        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(2, 3, figure=fig)
        
        # 3D end-effector 궤적 시각화
        ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
        ax_3d.plot(target_points[:, 0], target_points[:, 1], target_points[:, 2],
                color='blue', linestyle='--', label='Target')
        ax_3d.plot(user_points[:, 0], user_points[:, 1], user_points[:, 2],
                color='red', linestyle=':', label='User')
        ax_3d.plot(aligned_points[:, 0], aligned_points[:, 1], aligned_points[:, 2],
                color='green', linestyle='-', linewidth=2, label='Interpolated')
        
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title('End-Effector Trajectory')
        ax_3d.legend()

        # 시간 배열 생성
        target_time = np.arange(len(target_degrees))
        user_time = np.arange(len(user_degrees))
        aligned_time = np.arange(len(aligned_degrees))
        
        # 각 관절별 개별 그래프 생성 (오른쪽 2x2 그리드)
        joint_titles = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4']
        for idx, joint in enumerate(range(4)):
            row = idx // 2  # 행 인덱스
            col = (idx % 2) + 1  # 열 인덱스 (1부터 시작)
            
            ax = fig.add_subplot(gs[row, col])
            
            ax.plot(target_time, target_degrees[:, joint],
                color='blue', linestyle='--', label='Target')
            ax.plot(user_time, user_degrees[:, joint],
                color='red', linestyle=':', label='User')
            ax.plot(aligned_time, aligned_degrees[:, joint],
                color='green', linestyle='-', label='Interpolated')
            
            ax.set_title(joint_titles[joint])
            ax.set_xlabel('Time step')
            ax.set_ylabel('Angle (deg)')
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()

        # 생성된 궤적 저장
        self.save_generated_trajectory(generated_df, classification_result, generation_number)
        
        return generated_df

    def save_generated_trajectory(self, generated_df, classification_result, generation_number=1):
        """생성된 궤적을 지정된 형식으로 저장"""
        generation_dir = os.path.join(os.getcwd(), "generation_trajectory")
        os.makedirs(generation_dir, exist_ok=True)
        
        filename = f"generation_trajectory_{classification_result}_{generation_number}.txt"
        generation_path = os.path.join(generation_dir, filename)
        
        # 데이터프레임 생성 및 형식 맞추기
        num_points = len(generated_df)
        full_df = pd.DataFrame(index=range(num_points))
        
        # 기본 필드 설정
        full_df['r'] = 'm'
        full_df['sequence'] = range(num_points)
        full_df['timestamp'] = [i * 20 for i in range(num_points)]  # 20ms 간격
        
        # 각도, 엔드이펙터터의 데이터 설정
        full_df['deg'] = (generated_df['deg1'].round(3).astype(str) + '/' + 
                         generated_df['deg2'].round(3).astype(str) + '/' +
                         generated_df['deg3'].round(3).astype(str) + '/' +
                         generated_df['deg4'].round(3).astype(str))

        full_df['endpoint'] = (generated_df['x_end'].round(3).astype(str) + '/' + 
                             generated_df['y_end'].round(3).astype(str) + '/' + 
                             generated_df['z_end'].round(3).astype(str))
        
        # 열 순서 지정 및 저장
        columns = ['deg', 'endpoint']
        full_df = full_df[columns]
        
        full_df.to_csv(generation_path, index=False)
        return generation_path

def main():
    # 기본 디렉토리 설정
    base_dir = os.path.join(os.getcwd(), "data")
    
    try:
        # 객체 초기화
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
        target_trajectory, _ = analyzer.load_target_trajectory(trajectory_type)
        print(f"Selected Target Trajectory: {selected_golden}")
        
        # 궤적 비교 및 시각화
        print("\nCompare and visualize trajectories")
        generation_dir = os.path.join(os.getcwd(), "generation_trajectory")
        os.makedirs(generation_dir, exist_ok=True)
        existing_files = [f for f in os.listdir(generation_dir) 
                         if f.startswith(f"generation_trajectory_{trajectory_type}_")]
        generation_number = len(existing_files) + 1
        
        # 궤적 생성 및 저장
        generated_df = generator.compare_trajectories(
            target_df=target_trajectory,
            user_df=user_trajectory,
            classification_result=trajectory_type,
            generation_number=generation_number
        )
        
        print("\nProcessing completed!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Please check the data directory structure and model path")

if __name__ == "__main__":
    main()