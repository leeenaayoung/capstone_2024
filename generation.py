import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from trajectory_generation import TrajectoryGenerator
import os
import random
from classification import TrajectoryDataset, set_seed, device, train_classification_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def transform_dataset_for_lstm(classification_dataset, sequence_length=50):
   sequences = []
   targets = []
   
   for data, _ in classification_dataset:
       data_np = data.cpu().numpy()
       for i in range(len(data_np) - sequence_length):
           sequences.append(data_np[i:i + sequence_length])
           targets.append(data_np[i + sequence_length])
           
   return (torch.FloatTensor(sequences).to(device), 
           torch.FloatTensor(targets).to(device))

def train_trajectory_model(classification_dataset, movement_type, num_samples=10, epochs=100, batch_size=32):
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f"Using device: {device}")
   
   # classification_dataset을 LSTM용으로 변환
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
       
       torch.save(model.state_dict(), 'best_generation_model.pth')
   
   # 베스트 모델 로드
   model.load_state_dict(torch.load('best_generation_model.pth'))
   return model, history, classification_dataset.scaler

def main():
   base_dir = os.path.join(os.getcwd(), "data")
   dataset = TrajectoryDataset(base_path=base_dir)  # 데이터셋 로드
   generator = TrajectoryGenerator()
   
   try:
       # 분류 모델 먼저 학습
       print(f"Dataset size: {len(dataset)}")
       unique_labels = dataset.get_available_movement_types()
       print(f"Available movement types: {unique_labels}")
       
       # 분류 모델 학습
       model_params = {
           'input_dim': 21,
           'd_model': 64,
           'nhead': 2,
           'num_layers': 3,
           'batch_size': 32,
           'epochs': 100
       }
       
       classification_model, scaler = train_classification_model(dataset, unique_labels, model_params)
       
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
           device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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