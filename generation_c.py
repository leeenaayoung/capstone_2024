import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import random
from c_t import TrajectoryDataset, set_seed, device, TransformerModel, get_unique_labels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = StandardScaler()

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=False  # 단방향 학습습
        )

        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
   
class TrajectoryGenerator:
    """궤적 생성, 변환, 시각화를 담당하는 클래스"""
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
        """
        DTW를 적용하여 궤적을 정렬하고 보간
        
        Parameters:
            target: 타겟 궤적
            subject: 사용자 궤적
            interpolation_weight: 보간 가중치 (0: 사용자 궤적에 가깝게, 1: 타겟 궤적에 가깝게)
        """
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
        """
        타겟과 사용자 궤적을 비교하고 정렬된 궤적을 생성하여 시각화
        
        Parameters:
            target_df: 타겟 궤적 데이터프레임
            user_df: 사용자 궤적 데이터프레임
            save_animation: 애니메이션 저장 여부
        Returns:
            aligned_trajectory: DTW로 정렬된 궤적
        """
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

def transform_dataset_for_lstm(classification_dataset, sequence_length=50):
    # 한 번에 numpy 배열로 변환
    all_data = [data.cpu().numpy() for data, _ in classification_dataset]
    all_data = np.stack(all_data)
    
    sequences = []
    targets = []
    
    for data in all_data:
        seq = np.lib.stride_tricks.sliding_window_view(
            data, 
            (sequence_length + 1, data.shape[-1])
        ).reshape(-1, sequence_length + 1, data.shape[-1])
        
        sequences.append(seq[:, :-1, :])
        targets.append(seq[:, -1, :])
    
    sequences_array = np.concatenate(sequences)
    targets_array = np.concatenate(targets)
    
    return (torch.from_numpy(sequences_array).float().to(device), 
            torch.from_numpy(targets_array).float().to(device))

@torch.cuda.amp.autocast()  # 혼합 정밀도 학습
def train_trajectory_model(classification_dataset, movement_type, num_samples=10, epochs=100, batch_size=64):
    scaler = torch.cuda.amp.GradScaler()
    
    # 데이터셋 변환 및 분할
    sequences, targets = transform_dataset_for_lstm(classification_dataset)
   
    # 데이터 분할  
    dataset_size = len(sequences)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_sequences = sequences[train_indices]
    train_targets = targets[train_indices]
    val_sequences = sequences[val_indices]
    val_targets = targets[val_indices]
    
    # 데이터로더 생성
    train_dataset = torch.utils.data.TensorDataset(train_sequences, train_targets)
    val_dataset = torch.utils.data.TensorDataset(val_sequences, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 모델 초기화
    input_size = sequences.shape[-1]  # 특성 개수
    model = TrajectoryLSTM(input_size=input_size).to(device)
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # 학습 기록
    history = {'train_loss': [], 'val_loss': []}
    
    # 학습 루프
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        train_loss /= len(train_loader)
        
        # 검증
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        val_loss /= len(val_loader)
        
        # 손실 기록
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        torch.save(model.state_dict(), 'best_generation_model.pth')
    
    # 베스트 모델 로드
    model.load_state_dict(torch.load('best_generation_model.pth'))
    return model, history, classification_dataset.scaler

def main():
    # cuda 최적화화
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    base_dir = os.path.join(os.getcwd(), "data")
    dataset = TrajectoryDataset(base_path=base_dir)  # 데이터셋 로드
    generator = TrajectoryGenerator()
    
    try:
        # 데이터셋 정보 출력
        print(f"Dataset size: {len(dataset)}")
        unique_labels = get_unique_labels(base_path="data/all_data")
        print(f"Available movement types: {unique_labels}")
        
        # 모델 파라미터 설정
        model_params = {
            'input_dim': 21,
            'd_model': 32,
            'nhead': 2,
            'num_layers': 1
        }
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # 저장된 best model 불러오기
        classification_model = TransformerModel(
            input_dim=model_params['input_dim'],
            d_model=model_params['d_model'],
            nhead=model_params['nhead'],
            num_layers=model_params['num_layers'],
            num_classes=len(unique_labels)
        ).to(device)
        
        classification_model.load_state_dict(
            torch.load('best_model.pth', map_location=device)
        )
        classification_model.eval()  # 평가 모드로 설정
        
        print("\nTrajectory List", unique_labels)
        
        if unique_labels:
            movement_type = random.choice(unique_labels)
            print(f"\nCurrent Trajectory: {movement_type}")
            
            classification_dataset = dataset 

            # LSTM 모델 학습 - 같은 데이터셋 재사용
            print("\nTraining Start...")
            model, history, _ = train_trajectory_model(
                dataset, movement_type, num_samples=10)
            
            # 테스트용 궤적 로드
            print("\nLoad Test Trajectory...")
            target_traj, user_traj = dataset.load_random_trajectories(movement_type)
            
            # LSTM으로 예측
            print("\nTrajectory Prediction...")
            if len(dataset) > 0:
                sequence, _ = dataset[0]
                sequence = sequence.unsqueeze(0).to(device)
                
                model.eval()
                with torch.no_grad():
                    predicted = model(sequence)
                    predicted = predicted.cpu().numpy()
                    predicted = scaler.inverse_transform(predicted)
                    
                print("Prediction Position:", predicted[0, :3]) 
            
            # 시각화 비교
            print("\n궤적 비교 시각화...")
            aligned_trajectory = generator.compare_trajectories(target_traj, user_traj)
            
            # 학습 결과 출력
            print("\nTraining Results:")
            print(f"Final train loss: {history['train_loss'][-1]:.4f}")
            print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
            
        else:
            print("사용 가능한 동작 타입이 없습니다.")
            
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()