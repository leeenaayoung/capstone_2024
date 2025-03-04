import os
import random
import torch
import numpy as np
import pandas as pd
from dataloader import ClassificationDataset
from utils import preprocess_trajectory_data

##################################
# 궤적 분류기
##################################
class TrajectoryAnalyzer:
    def __init__(self, classification_model: str = "best_classification_model.pth", base_dir="data"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        c_base_path = "data/all_data"
        self.c_dataset = ClassificationDataset(c_base_path)  
        print("Available labels", self.c_dataset.unique_labels)

        self.trajectory_types = {i: label for i, label in enumerate(self.c_dataset.unique_labels)}
        # print("\nGenerated trajectory_types:", self.trajectory_types)

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
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
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
    
    # 타겟 궤적 하나로 수정
    def load_target_trajectory(self, trajectory_type: str, user_df=None):
        """ user_trajectory와 같은 타입의 target_trajectory 로드"""
        try:
            matching_files = [f for f in os.listdir(self.golden_dir) 
                            if f.startswith(trajectory_type) and f.endswith('.txt')]
            
            if not matching_files:
                # 매칭되는 파일이 없으면 오류 발생
                raise ValueError(f"From the golden_sample directory {trajectory_type} can't find the trajectory of the type")
            
            # 무작위 선택 대신 사용자 궤적의 분류 결과와 동일한 이름의 파일 선택
            if len(matching_files) == 1:
                selected_file = matching_files[0]
            else:
                # 여러 파일이 있는 경우, 파일명에 trajectory_type이 정확히 포함된 것을 우선 선택
                exact_matches = [f for f in matching_files if f.startswith(f"{trajectory_type}_")]
                if exact_matches:
                    selected_file = exact_matches[0]
                else:
                    # 정확한 매치가 없으면 첫 번째 파일 선택
                    selected_file = matching_files[0]
            
            file_path = os.path.join(self.golden_dir, selected_file)
            print(f"Using target trajectory: {selected_file}")
            
            # 선택된 파일 로드 및 전처리
            df = pd.read_csv(file_path, delimiter=',')
            _, preprocessed_df = preprocess_trajectory_data(df, scaler=self.c_dataset.scaler, return_raw=True)
            
            return preprocessed_df, selected_file
            
        except Exception as e:
            print(f"Error loading target trajectory: {str(e)}")
            raise
    # 기존
    # def load_target_trajectory(self, trajectory_type: str):
    #     """ user_trajectory와 같은 타입의 target_trajectory 로드"""
    #     try:
    #         matching_files = [f for f in os.listdir(self.golden_dir) 
    #                         if f.startswith(trajectory_type) and f.endswith('.txt')]
            
    #         if not matching_files:
    #             raise ValueError(f"From the golden_sample directory {trajectory_type} can't find the trajectory of the type")
            
    #         # 매칭되는 파일들 중 하나를 무작위로 선택(타겟 궤적 하나로 수정)
    #         selected_file = random.choice(matching_files)
    #         file_path = os.path.join(self.golden_dir, selected_file)
            
    #         # 선택된 파일 로드 및 전처리
    #         df = pd.read_csv(file_path, delimiter=',')
    #         _, preprocessed_df = preprocess_trajectory_data(df, scaler=self.c_dataset.scaler, return_raw=True)

            
    #         return preprocessed_df, selected_file
            
    #     except Exception as e:
    #         print(f"Error loading target trajectory: {str(e)}")
    #         raise

    def validate_input(df):
        if len(df) < 3:
            raise ValueError("Insufficient data points. At least 3 points are required.")
        if not all(col in df.columns for col in ['x_end', 'y_end', 'z_end']):
            raise ValueError("Input DataFrame must contain 'x_end', 'y_end', and 'z_end' columns.")
        
    def classify_trajectory_type(self, trajectory_type: str) -> str:
        """세부 궤적 유형을 주요 궤적 유형(line, arc, circle)으로 분류"""
        if any(t in trajectory_type for t in ['d_l', 'd_r']):
            return 'line'
        elif any(t in trajectory_type for t in ['v_45', 'v_90', 'v_135', 'v_180', 'h_u', 'h_d']):
            return 'arc'
        elif any(t in trajectory_type for t in ['clock_big', 'clock_t', 'clock_m', 'clock_b', 'clock_l', 'clock_r', 'counter_big', 'counter_t', 'counter_m', 'counter_b', 'counter_l', 'counter_r']):
            return 'circle'
        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")
    