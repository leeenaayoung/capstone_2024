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
    """관절 간 상관관계를 학습하는 트랜스포머 모델 (각도 + 각속도 포함)"""
    def __init__(self, input_dim=8, d_model=32, n_head=2, n_layers=4, dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim  # 입력 차원: 각도 4 + 각속도 4 
        self.d_model = d_model
        self.joint_embedding = nn.Linear(input_dim, d_model)  # 8 -> 32
        self.positional_encoding = PositionalEncoding(d_model)
        
        # 입력 정규화 및 드롭아웃
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        # Joint Attention 레이어 (2개로 감소)
        self.joint_attention_layers = nn.ModuleList([
            JointAttention(d_model) for _ in range(n_layers)
        ])
        
        # TransformerEncoderLayer 단순화 (n_head=2, n_layers=2)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 2,  # 간소화
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=n_layers
        )
        
        # 출력 레이어: 각도만 예측 (4차원 출력)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4)  # 각도 4 출력
        )
        
    def forward(self, x):
        # x: (batch_size, sequence_length, input_dim=8)
        x = self.joint_embedding(x)  # (batch_size, sequence_length, d_model)
        x = self.positional_encoding(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        joint_features = x
        for attention_layer in self.joint_attention_layers:
            joint_attention = attention_layer(joint_features)
            joint_features = joint_features + joint_attention  # Residual connection

        x = joint_features.transpose(0, 1)  # (sequence_length, batch_size, d_model)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch_size, sequence_length, d_model)
        
        output = self.output_layer(x)  # (batch_size, sequence_length, 4)
        return output
    
class TrajectoryDataset(Dataset):
    """궤적 데이터셋 클래스 (각도 + 각속도 포함)"""
    def __init__(self, trajectories):
        self.data = []
        for traj in trajectories:
            if isinstance(traj, pd.DataFrame):
                angles = traj[['deg1', 'deg2', 'deg3', 'deg4']].values
                velocities = traj[['degsec1', 'degsec2', 'degsec3', 'degsec4']].values
                combined = np.column_stack([angles, velocities])  # (sequence_length, 8)
                self.data.append(torch.FloatTensor(combined))
            else:
                self.data.append(torch.FloatTensor(traj))  # 이미 결합된 경우
    
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
        """학습 데이터 수집 (각도 + 각속도 포함)"""
        if self.analyzer is None:
            print("Analyzer not initialized. Cannot collect training data.")
            return []
            
        base_dir = data_dir or os.path.join(self.base_dir, "all_data")
        trajectories = []
        
        try:
            trajectory_files = [f for f in os.listdir(base_dir) if f.endswith('.txt')]
            
            if len(trajectory_files) > n_samples:
                trajectory_files = random.sample(trajectory_files, n_samples)
            
            print(f"Collecting data from {len(trajectory_files)} trajectory files...")
            
            for file_name in tqdm(trajectory_files, desc="Loading files"):
                file_path = os.path.join(base_dir, file_name)
                trajectory_type = None
                for type_name in ['d_', 'clock', 'counter', 'v_s', 'h_']:
                    if type_name in file_name.lower():
                        trajectory_type = type_name
                        break
                
                if trajectory_type:
                    try:
                        df, _ = self.analyzer.load_target_trajectory(trajectory_type)
                        
                        # 각도와 각속도 데이터 추출
                        angles = df[['deg1', 'deg2', 'deg3', 'deg4']].values
                        velocities = df[['degsec1', 'degsec2', 'degsec3', 'degsec4']].values
                        combined = np.column_stack([angles, velocities])  # (sequence_length, 8)
                        
                        if len(combined) >= 30:
                            trajectories.append(combined)
                    except Exception as e:
                        print(f"Error processing file {file_name}: {str(e)}")
            
            print(f"Successfully collected {len(trajectories)} trajectory data samples")
            if trajectories:
                lengths = [len(traj) for traj in trajectories]
                print(f"Trajectory statistics - Min: {min(lengths)}, Max: {max(lengths)}, Average: {np.mean(lengths):.2f}")
        
        except Exception as e:
            print(f"Error during data collection: {str(e)}")
        
        return trajectories
    
    def train_model(self, trajectories=None, epochs=100, batch_size=32, learning_rate=0.001):
        """관절 관계 모델 학습"""
        if trajectories is None:
            trajectories = self.collect_training_data()
            
        if not trajectories:
            print("No training data available. Skipping model training.")
            return False
        
        def collate_fn(batch):
            max_len = max(len(seq) for seq in batch)
            padded_batch = []
            for seq in batch:
                if len(seq) < max_len:
                    pad = torch.zeros((max_len - len(seq), seq.shape[1]))
                    padded_seq = torch.cat([seq, pad], dim=0)
                else:
                    padded_seq = seq[:max_len]
                padded_batch.append(padded_seq)
            return torch.stack(padded_batch)
        
        dataset = TrajectoryDataset(trajectories)
        train_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        self.model.train()
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        best_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        
        print(f"Starting joint relationship model training with {len(trajectories)} trajectories...")
        for epoch in range(epochs):
            epoch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            total_loss = 0.0
            
            for batch in epoch_progress:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # 입력: 각도 + 각속도 (8차원), 타겟: 각도만 (4차원)
                output = self.model(batch)  # (batch_size, sequence_length, 4)
                target = batch[:, :, :4]    # 타겟은 각도만
                
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                epoch_progress.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = self.model.state_dict().copy()
                best_epoch = epoch + 1
                print(f"New best model found at epoch {best_epoch} with loss: {best_loss:.4f}")
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            torch.save({
                'model_state_dict': best_model_state,
                'epoch': best_epoch,
                'loss': best_loss
            }, self.model_path)
            print(f"Best model saved (from epoch {best_epoch} with loss {best_loss:.4f})")
        else:
            print("Warning: No best model state found.")
        
        self.model.eval()
        return True