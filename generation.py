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

    # 기존 
    # def apply_dtw(self, target, subject, interpolation_weight=0.5):
    #     """DTW를 적용하여 궤적을 정렬하고 보간"""
    #     # 시간 정규화
    #     target_norm = self.normalize_time(target)
    #     subject_norm = self.normalize_time(subject)

    #     # 스무딩 적용
    #     target_smoothed = np.zeros_like(target_norm)
    #     subject_smoothed = np.zeros_like(subject_norm)
        
    #     for i in range(target_norm.shape[1]):
    #         target_smoothed[:, i] = self.smooth_data(target_norm[:, i])
    #         subject_smoothed[:, i] = self.smooth_data(subject_norm[:, i])

    #     # DTW 거리 및 경로 계산
    #     distance, path = fastdtw(target_smoothed, subject_smoothed, dist=euclidean)
    #     path = np.array(path)

    #     # 매칭된 포인트들 추출
    #     target_matched = target_smoothed[path[:, 0]]
    #     subject_matched = subject_smoothed[path[:, 1]]

    #     # 보간된 궤적 생성
    #     interpolated = (target_matched * interpolation_weight + 
    #                    subject_matched * (1 - interpolation_weight))

    #     # 결과 궤적을 원본 길이로 리샘플링
    #     return self.normalize_time(interpolated, num_points=len(target))

    def apply_dtw(self, target, subject):
        """DTW를 적용하여 궤적을 정렬하고 보간"""
        # 시간 정규화
        target_norm = self.normalize_time(target)
        subject_norm = self.normalize_time(subject)

        max_dims = max(target_norm.shape[1], subject_norm.shape[1])
    
        if target_norm.shape[1] < max_dims:
            pad_width = ((0, 0), (0, max_dims - target_norm.shape[1]))
            target_norm = np.pad(target_norm, pad_width, mode='constant', constant_values=0)
        elif subject_norm.shape[1] < max_dims:
            pad_width = ((0, 0), (0, max_dims - subject_norm.shape[1]))
            subject_norm = np.pad(subject_norm, pad_width, mode='constant', constant_values=0)

        # 스무딩 적용
        target_smoothed = np.zeros_like(target_norm)
        subject_smoothed = np.zeros_like(subject_norm)
        
        for i in range(target_norm.shape[1]):
            target_smoothed[:, i] = self.smooth_data(target_norm[:, i])
            subject_smoothed[:, i] = self.smooth_data(subject_norm[:, i])

        # DTW 거리 및 경로 계산
        distance, path = fastdtw(target_smoothed, subject_smoothed, dist=euclidean)
        path = np.array(path)

        # 매칭된 포인트들 추출
        target_matched = target_smoothed[path[:, 0]]
        subject_matched = subject_smoothed[path[:, 1]]

        # 보간 가중치 결정
        interpolated_results = []
    
        # 각 가중치에 대해 보간 수행
        for weight in np.arange(0.1, 1.0, 0.1):
            # 보간된 궤적 생성
            interpolated = (target_matched * weight + 
                        subject_matched * (1 - weight))
            
            # 결과 궤적을 원본 길이로 리샘플링
            resampled = self.normalize_time(interpolated, num_points=len(target))
            interpolated_results.append(resampled)

        return interpolated_results

    def compare_trajectories(self, target_df, user_df, classification_result, generation_number=1):
        """타겟과 사용자 궤적을 비교하고 정렬된 궤적을 생성하여 시각화"""
        # end-effector 위치 추출
        target_points = target_df[['x_end', 'y_end', 'z_end']].values
        user_points = user_df[['x_end', 'y_end', 'z_end']].values
        
        # degree 값 추출
        target_degrees = target_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_degrees = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        
        # DTW를 사용하여 정렬된 궤적 생성 (end-effector 위치 기반)
        # aligned_trajectory = self.apply_dtw(target_points, user_points)
        
        # 동일한 비율로 degree 값도 정렬
        # aligned_degrees = self.apply_dtw(target_degrees, user_degrees)

        # DTW를 사용하여 정렬된 궤적들 생성 (end-effector 위치 기반)
        aligned_trajectories = self.apply_dtw(target_points, user_points)
        
        # 동일한 비율로 degree 값도 정렬
        aligned_degrees_list = self.apply_dtw(target_degrees, user_degrees)

        # 수정
        # 각 가중치별 결과에 대해 처리
        generated_dfs = []
        for idx, (aligned_trajectory, aligned_degrees) in enumerate(zip(aligned_trajectories, aligned_degrees_list)):
            weight = 0.1 * (idx + 1)
            
            # 생성된 궤적 저장 (end-effector 위치와 degree 값 모두)
            generated_df = pd.DataFrame(
                np.hstack([aligned_trajectory, aligned_degrees]),
                columns=['x_end', 'y_end', 'z_end', 'deg1', 'deg2', 'deg3', 'deg4']
            )
            
            # 시각화 (end-effector 궤적만)
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
                    'g-', label=f'Aligned Trajectory (weight={weight:.1f})', linewidth=2)

            # 그래프 설정
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            ax.set_zlabel('Z Axis')
            ax.set_title(f'Trajectory Alignment using DTW (weight={weight:.1f})')
            ax.view_init(10, 90)
            ax.legend(bbox_to_anchor=(1.15, 1), loc='upper right')
            ax.grid(True)

            # 모든 궤적을 포함하도록 축 범위 설정
            all_points = np.vstack([target_points, user_points, aligned_trajectory])
            margin = 10 
            ax.set_xlim([min(all_points[:, 0]) - margin, max(all_points[:, 0]) + margin])
            ax.set_ylim([min(all_points[:, 1]) - margin, max(all_points[:, 1]) + margin])
            ax.set_zlim([min(all_points[:, 2]) - margin, max(all_points[:, 2]) + margin])

            plt.tight_layout()
            plt.show()
            
            # 생성된 궤적 저장
            self.save_generated_trajectory(generated_df, classification_result, 
                                        f"{generation_number}_w{weight:.1f}")
            
            generated_dfs.append(generated_df)
        
        return generated_dfs
        
        # 생성된 궤적 저장 (end-effector 위치와 degree 값 모두)
        # 기존
        # generated_df = pd.DataFrame(
        #     np.hstack([aligned_trajectory, aligned_degrees]),
        #     columns=['x_end', 'y_end', 'z_end', 'deg1', 'deg2', 'deg3', 'deg4']
        # )
        
        # # 시각화 (end-effector 궤적만)
        # fig = plt.figure(figsize=(12, 8))
        # ax = fig.add_subplot(111, projection='3d')
        
        # # 타겟 궤적
        # ax.plot(target_points[:, 0], target_points[:, 1], target_points[:, 2],
        #         'b--', label='Target Trajectory')
        
        # # 원본 사용자 궤적
        # ax.plot(user_points[:, 0], user_points[:, 1], user_points[:, 2],
        #         'r--', label='Original Subject Trajectory')
        
        # # 정렬된 궤적
        # ax.plot(aligned_trajectory[:, 0], aligned_trajectory[:, 1], aligned_trajectory[:, 2],
        #         'g-', label='Aligned Subject Trajectory', linewidth=2)

        # # 그래프 설정
        # ax.set_xlabel('X Axis')
        # ax.set_ylabel('Y Axis')
        # ax.set_zlabel('Z Axis')
        # ax.set_title('Trajectory Alignment using DTW')
        # ax.view_init(10, 90)
        # ax.legend(bbox_to_anchor=(1.15, 1), loc='upper right')
        # ax.grid(True)

        # # 모든 궤적을 포함하도록 축 범위 설정
        # all_points = np.vstack([target_points, user_points, aligned_trajectory])
        # margin = 10 
        # ax.set_xlim([min(all_points[:, 0]) - margin, max(all_points[:, 0]) + margin])
        # ax.set_ylim([min(all_points[:, 1]) - margin, max(all_points[:, 1]) + margin])
        # ax.set_zlim([min(all_points[:, 2]) - margin, max(all_points[:, 2]) + margin])

        # plt.tight_layout()
        # plt.show()
        
        # # 생성된 궤적 저장
        # self.save_generated_trajectory(generated_df, classification_result, generation_number)
        
        # return generated_df

    def save_generated_trajectory(self, generated_df, classification_result, generation_number=1):
        """생성된 궤적을 지정된 형식으로 저장"""
        # 저장 디렉토리 및 파일명 설정
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
        full_df['timestamp'] = [i * 30 for i in range(num_points)]  # 30ms 간격
        
        # degree 값 설정
        full_df['deg'] = (generated_df['deg1'].round(3).astype(str) + '/' + 
                        generated_df['deg2'].round(3).astype(str) + '/' +
                        generated_df['deg3'].round(3).astype(str) + '/' +
                        generated_df['deg4'].round(3).astype(str))
        
        # deg/sec 설정
        full_df['deg/sec'] = '0/0/0/0'  # 기본값 설정
        
        # endpoint 설정
        full_df['endpoint'] = (generated_df['x_end'].round(3).astype(str) + '/' + 
                            generated_df['y_end'].round(3).astype(str) + '/' + 
                            generated_df['z_end'].round(3).astype(str))
        
        # 나머지 필드 설정
        full_df['mA'] = '0/0/0/0'
        full_df['grip/rotation'] = '0/0'
        full_df['torque'] = '-2200/1800/-14000'
        full_df['force'] = '-800/310/690'
        full_df['ori'] = '1788/-104/19'
        full_df['#'] = '#'
        
        # 열 순서 지정
        column_order = ['r', 'sequence', 'timestamp', 'deg', 'deg/sec', 'mA', 
                        'endpoint', 'grip/rotation', 'torque', 'force', 'ori', '#']
        full_df = full_df[column_order]
        
        # 파일 저장
        full_df.to_csv(generation_path, index=False)
        # print(f"\nGenerated trajectory saved to: {generation_path}")
        
        return generation_path

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
        target_trajectory, _ = analyzer.load_target_trajectory(trajectory_type)
        print(f"Selected Target Trajectory: {selected_golden}")
        
        # 궤적 비교 및 시각화
        print("\nCompare and visualize trajectories")
        generation_dir = os.path.join(os.getcwd(), "generation_trajectory")
        os.makedirs(generation_dir, exist_ok=True)
        # existing_files = [f for f in os.listdir(generation_dir) 
        #                  if f.startswith(f"generation_trajectory_{trajectory_type}_")]
        # generation_number = len(existing_files) + 1
        
        # 궤적 생성 및 저장
        # 기존
        # generated_df = generator.compare_trajectories(
        #     target_df=target_trajectory,
        #     user_df=user_trajectory,
        #     classification_result=trajectory_type,
        #     generation_number=generation_number
        # )
        # 각 가중치별로 궤적 생성 및 저장
        generated_dfs = generator.compare_trajectories(
            target_df=target_trajectory,
            user_df=user_trajectory,
            classification_result=trajectory_type,
            generation_number=1  # 또는 원하는 번호
        )
        
        print("\nProcessing completed!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Please check the data directory structure and model path")

if __name__ == "__main__":
    main()