import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from trajectory_generation import TrajectoryAnalyzer
from trajectory_generation import TrajectoryGenerator
import os
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrajectoryDataset(Dataset):
    def __init__(self, df, sequence_length=50):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.features = ['x_end', 'y_end', 'z_end', 'yaw', 'pitch', 'roll',
                        'deg1', 'deg2', 'deg3', 'deg4',
                        'torque1', 'torque2', 'torque3', 'force1', 'force2', 'force3']
        data = df[self.features].values
        
        # 데이터 스케일링
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data)
        
        # 시퀀스 생성
        self.sequences = []
        self.targets = []
        
        for i in range(len(scaled_data) - sequence_length):
            self.sequences.append(scaled_data[i:i + sequence_length])
            self.targets.append(scaled_data[i + sequence_length])
            
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.sequences[idx]).to(self.device), 
                torch.FloatTensor(self.targets[idx]).to(self.device))

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(TrajectoryLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_size)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :]) 

def train_trajectory_model(analyzer, movement_type, num_samples=10, epochs=100, batch_size=32):
    """
    특정 동작 타입의 golden sample들로 LSTM 모델 학습
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Golden sample 수집
    golden_samples = []
    for _ in range(num_samples):
        try:
            target_traj, _ = analyzer.get_random_trajectory_by_type(
                analyzer.golden_dir, movement_type
            )
            golden_samples.append(target_traj)
        except Exception as e:
            print(f"Error collecting sample: {str(e)}")
    
    if not golden_samples:
        raise ValueError("No golden samples could be collected")
    
    # 데이터셋 준비
    combined_df = pd.concat(golden_samples, ignore_index=True)
    dataset = TrajectoryDataset(combined_df)

    # 트레이닝/검증 분할
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size)
    
    # 모델 초기화
    input_size = len(dataset.features)
    model = TrajectoryLSTM(input_size=input_size)
    model = model.to(device)
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Early stopping 설정
    # best_val_loss = float('inf')
    # patience = 5
    # patience_counter = 0
    
    # 학습 기록
    history = {'train_loss': [], 'val_loss': []}
    
    # 학습 루프
    for epoch in range(epochs):
        # 트레이닝
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
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
        
        # # Early stopping 체크
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     patience_counter = 0
        #     # 베스트 모델 저장
        torch.save(model.state_dict(), 'best_model.pth')
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print('Early stopping!')
        #         break
    
    # 베스트 모델 로드
    model.load_state_dict(torch.load('best_model.pth'))
    return model, history, dataset.scaler

def main():
    base_dir = os.path.join(os.getcwd(), "data")
    analyzer = TrajectoryAnalyzer(base_dir)
    generator = TrajectoryGenerator()
    
    try:
        # 사용 가능한 동작 타입 출력
        movement_types = analyzer.get_available_movement_types()
        print("\nTrajectory List", movement_types)
        
        if movement_types:
            movement_type = random.choice(movement_types)
            print(f"\nCurrent Trajectory: {movement_type}")
            
            # LSTM 모델 학습
            print("\nTraining Start...")
            model, history, scaler = train_trajectory_model(
                analyzer, movement_type, num_samples=5)
            
            # 테스트용 궤적 로드
            print("\nLoad Test Trajectory...")
            target_traj, user_traj = analyzer.load_random_trajectories(movement_type)
            
            # LSTM으로 예측
            print("\nTrajectory Predictioin...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dataset = TrajectoryDataset(target_traj)
            if len(dataset) > 0:
                sequence, _ = dataset[0]
                sequence = sequence.unsqueeze(0).to(device)
                
                model.eval()
                with torch.no_grad():
                    predicted = model(sequence)
                    predicted = predicted.cpu().numpy()
                    predicted = scaler.inverse_transform(predicted)
                    
                print("Prediction Position:", predicted[0, :3])  # x, y, z 좌표만 출력
            
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