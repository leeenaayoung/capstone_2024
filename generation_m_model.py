import numpy as np
import pandas as pd
import torch
import os
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from analyzer import TrajectoryAnalyzer
from utils import calculate_end_effector_position

class MultiHeadJointAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        x = x + self.dropout(attn_output)
        return self.norm(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class JointTrajectoryTransformer(nn.Module):
    def __init__(self, n_joints=4, d_model=128, n_head=8, n_layers=6, dropout=0.2, n_velocities=4):
        super().__init__()
        # 입력 차원: 각도(n_joints) + 각속도(n_velocities) + 엔드이펙터 위치(3) + timestamp(1)
        self.input_dim = n_joints + n_velocities + 3 + 1  # 4 + 4 + 3 + 1 = 12
        self.joint_embedding = nn.Linear(self.input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length=5000)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        
        self.joint_attention_layers = nn.ModuleList([
            MultiHeadJointAttention(d_model, n_head) for _ in range(n_layers)
        ])
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,  # batch_first=True로 설정
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=n_layers)
        
        self.output_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_joints)
        )
    
    def forward(self, x, velocities, timestamps):
        batch_size, seq_len, _ = x.size()
        combined_input = torch.cat([x, velocities, timestamps], dim=-1)  # (batch_size, seq_len, 12)
        
        # 동적 엔드이펙터 위치 계산
        positions = torch.zeros(batch_size, seq_len, 3).to(x.device)
        for i in range(seq_len):
            positions[:, i, :] = torch.tensor(calculate_end_effector_position(x[0, i, :4].cpu().numpy())).to(x.device)
        combined_input = torch.cat([combined_input, positions], dim=-1)  # (batch_size, seq_len, 15)
        
        x = self.joint_embedding(combined_input)  # (batch_size, seq_len, d_model)
        x = self.positional_encoding(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        joint_features = x
        for attention_layer in self.joint_attention_layers:
            joint_attention = attention_layer(joint_features)
            joint_features = joint_features + joint_attention

        x = joint_features.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        x = self.output_norm(x)
        x = self.output_dropout(x)
        output = self.output_layer(x)
        
        return output
# TrajectoryDataset 클래스 (Timestamp 추가)
class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        self.data = []
        for traj in trajectories:
            angles = traj[:, :4]  # 각도
            velocities = np.gradient(angles, axis=0)  # 각속도
            positions = np.array([calculate_end_effector_position(deg) for deg in angles])  # 엔드이펙터
            # Timestamp 생성 (단순화된 시간 스탬프: 0부터 시작, 일정한 간격)
            timestamps = np.arange(len(angles)).reshape(-1, 1) / len(angles)  # 정규화된 시간 (0~1)
            combined = np.concatenate([angles, velocities, positions, timestamps], axis=1)
            self.data.append(torch.FloatTensor(combined.squeeze()))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# GenerationModel 클래스 (전체 수정)
class GenerationModel:
    def __init__(self, base_dir=None, model_save_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_dir = base_dir or os.path.join(os.getcwd(), "data")

        if model_save_path:
            self.model_path = model_save_path
        else:
            self.model_path = os.path.join(os.getcwd(), "best_generation_model.pth")
        
        # 개선된 모델 초기화
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
                        angles = df[['deg1', 'deg2', 'deg3', 'deg4']].values
                        if len(angles) >= 30:
                            trajectories.append(angles)
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
        """관절 관계 모델 학습 (시간적 패턴 반영)"""
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
        smoothness_criterion = nn.MSELoss()  # 시간적 연속성 손실
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
                
                # 데이터 분리
                angles = batch[:, :, :4]  # 각도
                velocities = batch[:, :, 4:8]  # 각속도
                timestamp = batch[:, :, 8:9]  # timestamp
                positions = batch[:, :, 9:12]  # 엔드이펙터 위치
                
                # 모델 예측
                output_angles = self.model(angles, velocities, timestamp)
                
                # 손실 계산
                angle_loss = criterion(output_angles, angles)  # 각도 예측 손실
                smoothness_loss = smoothness_criterion(output_angles[:, 1:, :], output_angles[:, :-1, :])  # 시간적 연속성 손실
                
                total_loss = angle_loss + 0.1 * smoothness_loss  # 시간적 연속성 손실 가중치
                total_loss.backward()
                optimizer.step()
                
                total_loss += total_loss.item()
                epoch_progress.set_postfix(loss=f"{total_loss.item():.4f}")
            
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
