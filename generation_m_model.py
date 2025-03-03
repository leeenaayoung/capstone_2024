import numpy as np
import pandas as pd
import torch
import os
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from analyzer import TrajectoryAnalyzer

class JointAttention(nn.Module):
    """관절 간의 관계를 학습하는 Self-Attention 모듈"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model).float())
        attention = torch.softmax(scores, dim=-1)
        
        return torch.matmul(attention, V)
    
class PositionalEncoding(nn.Module):
    """시간 정보를 인코딩"""
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class JointTrajectoryTransformer(nn.Module):
    """관절 간 상관관계를 학습하는 트랜스포머 모델"""
    def __init__(self, n_joints=4, d_model=64, n_head=4, n_layers=4, dropout=0.2):
        super().__init__()
        
        self.joint_embedding = nn.Linear(n_joints, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # 입력 레이어
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.joint_attention_layers = nn.ModuleList([
            JointAttention(d_model) for _ in range(n_layers)
        ])
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=n_layers
        )
        
        # 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, n_joints)
        )
        
    def forward(self, x):
        x = self.joint_embedding(x)
        x = self.positional_encoding(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        joint_features = x
        for attention_layer in self.joint_attention_layers:
            joint_attention = attention_layer(joint_features)
            joint_features = joint_features + joint_attention

        x = joint_features.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        
        # 출력 생성
        output = self.output_layer(x)
        
        return output

class TrajectoryDataset(Dataset):
    """궤적 데이터셋 클래스"""
    def __init__(self, trajectories):
        self.data = [torch.FloatTensor(traj) for traj in trajectories]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
class GenerationModel:
    """모델 훈련 전용 클래스"""
    def __init__(self, base_dir=None, model_save_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_dir = base_dir or os.path.join(os.getcwd(), "data")

        if model_save_path:
            self.model_path = model_save_path
        else:
            self.model_path = os.path.join(os.getcwd(), "best_generation_model.pth")
        
        # 모델 초기화
        self.model = JointTrajectoryTransformer().to(self.device)
        
        # 분석기 초기화
        try:
            self.analyzer = TrajectoryAnalyzer(
                classification_model="best_classification_model.pth",
                base_dir=self.base_dir
            )
        except Exception as e:
            print(f"Analyzer initialization error: {str(e)}")
            self.analyzer = None
    
    def collect_training_data(self, data_dir=None, n_samples=100):
        """학습 데이터 수집"""
        if self.analyzer is None:
            print("Analyzer not initialized. Cannot collect training data.")
            return []
            
        # 데이터 디렉토리 설정
        base_dir = data_dir or os.path.join(self.base_dir, "all_data")
        trajectories = []
        
        try:
            trajectory_files = [f for f in os.listdir(base_dir) if f.endswith('.txt')]
            
            if len(trajectory_files) > n_samples:
                trajectory_files = random.sample(trajectory_files, n_samples)
            
            print(f"Collecting data from {len(trajectory_files)} trajectory files...")
            
            for file_name in tqdm(trajectory_files, desc="Loading files"):
                file_path = os.path.join(base_dir, file_name)
                
                # 궤적 유형 추출
                trajectory_type = None
                for type_name in ['d_', 'clock', 'counter', 'v_s', 'h_']:
                    if type_name in file_name.lower():
                        trajectory_type = type_name
                        break
                
                if trajectory_type:
                    try:
                        # TrajectoryAnalyzer에서 궤적 로드
                        df, _ = self.analyzer.load_target_trajectory(trajectory_type)
                        
                        # 각도 데이터 추출
                        angles = df[['deg1', 'deg2', 'deg3', 'deg4']].values
                        
                        # 궤적이 너무 짧지 않은 경우에만 추가
                        if len(angles) >= 30:
                            trajectories.append(angles)
                    except Exception as e:
                        print(f"Error processing file {file_name}: {str(e)}")
            
            print(f"Successfully collected {len(trajectories)} trajectory data samples")

            # 데이터 통계 출력
            if trajectories:
                lengths = [len(traj) for traj in trajectories]
                print(f"Trajectory statistics - Min: {min(lengths)}, Max: {max(lengths)}, Average: {np.mean(lengths):.2f}")
    
        except Exception as e:
            print(f"Error during data collection: {str(e)}")
        
        return trajectories
    
    def train_model(self, trajectories=None, epochs=100, batch_size=32, learning_rate=0.001):
        """관절 관계 모델 학습"""
        # 데이터가 제공되지 않은 경우 자동 수집
        if trajectories is None:
            trajectories = self.collect_training_data()
            
        if not trajectories:
            print("No training data available. Skipping model training.")
            return False
        
        def collate_fn(batch):
            # 가장 긴 시퀀스 길이 찾기
            max_len = max(len(seq) for seq in batch)
            
            # 패딩된 배치 생성
            padded_batch = []
            for seq in batch:
                if len(seq) < max_len:
                    # 마지막 벡터로 패딩
                    pad = torch.zeros((max_len - len(seq), seq.shape[1]))
                    padded_seq = torch.cat([seq, pad], dim=0)
                else:
                    padded_seq = seq[:max_len]
                padded_batch.append(padded_seq)
            
            return torch.stack(padded_batch)
            
        # 데이터셋 및 데이터로더 생성
        dataset = TrajectoryDataset(trajectories)
        train_loader = DataLoader(
                            dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            collate_fn=collate_fn
                        )
        # 모델을 학습 모드로 설정
        self.model.train()
        
        # 손실 함수 및 옵티마이저 설정
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 최적의 모델 추적을 위한 변수
        best_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        
        # 학습 루프
        print(f"Starting joint relationship model training with {len(trajectories)} trajectories...")
        for epoch in range(epochs):
            # 각 에폭마다 새로운 tqdm 바 생성
            epoch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            total_loss = 0.0
            
            for batch in epoch_progress:
                batch = batch.to(self.device)
                
                # 순전파
                optimizer.zero_grad()
                output = self.model(batch)
                
                # 손실 계산 - 관절 간 관계를 학습하기 위해 원본 궤적을 예측하도록 학습
                loss = criterion(output, batch)
                
                # 역전파 및 최적화
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                # tqdm 바에 현재 손실 표시
                epoch_progress.set_postfix(loss=f"{loss.item():.4f}")
            
            # 에포크별 평균 손실 계산
            avg_loss = total_loss/len(train_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')
            
            # 최적의 모델 상태 저장
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = self.model.state_dict().copy()
                best_epoch = epoch + 1
                print(f"New best model found at epoch {best_epoch} with loss: {best_loss:.4f}")
        
        # 최적의 모델 복원 및 저장
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            torch.save({
                        'model_state_dict': best_model_state,
                        'epoch': best_epoch,
                        'loss': best_loss
                    }, self.model_path)
            
            print(f"Best model saved (from epoch {best_epoch} with loss {best_loss:.4f})")
        else:
            print("Warning: No best model state found. This shouldn't happen.")
        
        # 모델을 다시 평가 모드로 설정
        self.model.eval()
        return True