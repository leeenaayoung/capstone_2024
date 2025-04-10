import torch
import torch.nn as nn
import math
import random
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import calculate_end_effector_position
from analyzer import TrajectoryAnalyzer
from endeffector_model import TrajectoryTransformer

def main():
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    model_path = "best_trajectory_transformer.pth"  # 학습된 모델 경로
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 분석기 초기화
    analyzer = TrajectoryAnalyzer(
        classification_model="best_classification_model.pth",
        base_dir=base_dir
    )

    # 사용자 궤적 파일 랜덤 선택
    non_golden_dir = os.path.join(data_dir, "non_golden_sample")
    non_golden_files = [f for f in os.listdir(non_golden_dir) if f.endswith('.txt')]
    
    if not non_golden_files:
        print("사용자 궤적 파일을 찾을 수 없습니다.")
        return
    
    # 랜덤하게 하나의 사용자 궤적 선택
    selected_file = random.choice(non_golden_files)
    print(f"선택된 사용자 궤적: {selected_file}")

    # 사용자 궤적 로드 및 분류
    user_path = os.path.join(non_golden_dir, selected_file)
    user_df, trajectory_type = analyzer.load_user_trajectory(user_path)
    
    # 해당 타입의 타겟 궤적 찾기
    golden_dir = os.path.join(data_dir, "golden_sample")
    golden_files = [f for f in os.listdir(golden_dir) if trajectory_type in f and f.endswith('.txt')]
    
    if not golden_files:
        print(f"{trajectory_type} 타입의 타겟 궤적을 찾을 수 없습니다.")
        return
    
    # 타겟 궤적 로드
    target_file = golden_files[0]
    print(f"매칭된 타겟 궤적: {target_file}")
    target_path = os.path.join(golden_dir, target_file)
    target_df, _ = analyzer.load_user_trajectory(target_path)

    # 모델 초기화에 필요한 입력 차원 설정 (7: 3개 위치 + 4개 관절 각도)
    input_dim = 7
    
    # 학습된 모델 로드
    print(f"Loading pre-trained model from {model_path}...")
    
    # 모델 초기화 - 올바른 매개변수 순서와 타입으로 전달
    model = TrajectoryTransformer(
        input_dim=input_dim,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_length=100,
        num_trajectory_types=5
    )
    
    # 학습된 모델 가중치 로드
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    
    # 궤적 데이터 준비 - 이미 전처리된 데이터를 사용
    target_data = prepare_model_input(target_df, device)
    user_data = prepare_model_input(user_df, device)
    
    # 궤적 타입 인덱스 설정
    type_to_idx = {
        'd_': 0,      # 대각선
        'clock': 1,   # 시계 방향
        'counter': 2, # 반시계 방향
        'v_': 3,      # 수직
        'h_': 4       # 수평
    }
    traj_type_idx = type_to_idx.get(trajectory_type, 0)
    traj_type_tensor = torch.tensor([traj_type_idx], device=device)
    
    # 보간 가중치 설정
    weight = 0.5
    print(f"Generating trajectory with weight {weight}...")
    
    # 모델을 사용한 궤적 보간
    interpolated_trajectory = model.interpolate_trajectories(
        target_data, 
        user_data, 
        interpolate_weight=weight,
        traj_type=traj_type_idx
    )
    
    # 출력 형식 변환
    interpolated_ee = interpolated_trajectory.cpu().numpy()
    
    # 원본 궤적 추출 (엔드이펙터 위치)
    target_ee = extract_endpoint_positions(target_df)
    user_ee = extract_endpoint_positions(user_df)
    
    # 결과 시각화
    print("Visualizing results...")
    visualize_trajectories(
        target_ee, 
        user_ee, 
        interpolated_ee,
        trajectory_type
    )
    
    print("Done!")

def prepare_model_input(df, device):
    """데이터프레임을 모델 입력 형식으로 변환"""
    # 관절 각도 추출
    joint_angles = df[['deg1', 'deg2', 'deg3', 'deg4']].values
    
    # 엔드이펙터 위치 계산
    endpoints = np.array([calculate_end_effector_position(deg) for deg in joint_angles]) * 1000
    
    # 위치와 각도 결합
    features = np.hstack((endpoints, joint_angles))
    
    # 텐서로 변환
    return torch.FloatTensor(features).to(device)

def extract_endpoint_positions(df):
    """데이터프레임에서 엔드이펙터 위치 추출"""
    joint_angles = df[['deg1', 'deg2', 'deg3', 'deg4']].values
    endpoints = np.array([calculate_end_effector_position(deg) for deg in joint_angles]) * 1000
    return endpoints

def visualize_trajectories(target_ee, user_ee, interpolated_ee, trajectory_type):
    """보간된 궤적 시각화"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(target_ee[:, 0], target_ee[:, 1], target_ee[:, 2], 
            color='blue', linewidth=2, label='Target Trajectory')
    
    ax.plot(user_ee[:, 0], user_ee[:, 1], user_ee[:, 2], 
            color='red', linewidth=2, label='User Trajectory')
    
    ax.plot(interpolated_ee[:, 0], interpolated_ee[:, 1], interpolated_ee[:, 2], 
            color='green', linewidth=3, linestyle='-', 
            label='Interpolated (w=0.5)')
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title(f'{trajectory_type.capitalize()} Trajectory Interpolation')
    ax.legend(loc='best')
    
    # 좌표축 비율 조정
    all_points = np.vstack([target_ee, user_ee, interpolated_ee])
    max_range = np.max([np.ptp(all_points[:, 0]), np.ptp(all_points[:, 1]), np.ptp(all_points[:, 2])]) / 2.0
    mid_x, mid_y, mid_z = np.mean(all_points, axis=0)
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    main()