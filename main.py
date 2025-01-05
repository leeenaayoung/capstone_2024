import random
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
from models import TransformerModel, LSTMModel, preprocess_trajectory_data

# non_golden_sample에서 무작위로 궤적을 선택하는 함수
def load_random_trajectory_from_non_golden(non_golden_dir="data/non_golden_sample"):
    """non_golden_sample에서 하나의 궤적을 무작위로 로드"""
    trajectory_files = [f for f in os.listdir(non_golden_dir) if f.endswith('.txt')]
    
    if not trajectory_files:
        raise ValueError(f"non_golden_sample에 궤적 파일이 없습니다.")
    
    selected_file = random.choice(trajectory_files)
    file_path = os.path.join(non_golden_dir, selected_file)
    
    # CSV 파일 읽기
    trajectory_data = pd.read_csv(file_path, delimiter=',')
    
    return trajectory_data, selected_file

# 모델 로딩 함수
def load_model(model_path, model_class, input_dim, d_model, nhead, num_layers, num_classes):
    """주어진 경로에서 모델을 로딩"""
    model = model_class(input_dim, d_model, nhead, num_layers, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 평가 모드로 설정
    return model

# 분류 모델을 사용하여 예측을 수행
def classify_trajectory(model, trajectory_tensor):
    with torch.no_grad():  # 평가 모드에서 gradient 계산 방지
        output = model(trajectory_tensor)
        _, predicted = torch.max(output, 1)
    return predicted

# 생성 모델을 사용하여 궤적을 생성
def generate_trajectory(model, trajectory_tensor):
    with torch.no_grad():
        generated_trajectory = model(trajectory_tensor)
    return generated_trajectory

# 메인 실행
def main(non_golden_dir="data/non_golden_sample", classification_model_path="best_model.pth", generation_model_path="best_generation_model.pth"):
    # non_golden_sample에서 궤적 무작위 선택
    trajectory_data, selected_file = load_random_trajectory_from_non_golden(non_golden_dir)
    print(f"선택된 궤적 파일: {selected_file}")

    # 궤적을 모델에 맞게 전처리
    trajectory_tensor = preprocess_trajectory_data(trajectory_data)

    input_dim = 21  # 전처리 후 실제 피처 수
    d_model = 32
    nhead = 2
    num_layers = 1
    num_classes = 20  # 실제 클래스 수에 맞게 설정

    # 분류 모델 로드 및 예측
    classification_model = load_model(classification_model_path, TransformerModel, input_dim, d_model, nhead, num_layers, num_classes)
    predicted_class = classify_trajectory(classification_model, trajectory_tensor)
    print(f"예측된 분류: {predicted_class.item()}")

    # 생성 모델 로드 및 생성
    generation_model = load_model(generation_model_path, LSTMModel)
    generated_trajectory = generate_trajectory(generation_model, trajectory_tensor)
    print(f"생성된 궤적: {generated_trajectory}")

if __name__ == "__main__":
    main(non_golden_dir="data/non_golden_sample", 
         classification_model_path="best_classification_model.pth", 
         generation_model_path="best_generation_model.pth")
