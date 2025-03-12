import numpy as np
import pandas as pd
import torch
import os
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from analyzer import TrajectoryAnalyzer
from utils import calculate_end_effector_position, preprocess_trajectory_data

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

class JointCorrelationModule(nn.Module):
    def __init__(self, n_joints=4, d_model=64):
        super().__init__()
        self.joint_pair_embedding = nn.Linear(2, d_model // 2)
        num_pairs = n_joints * (n_joints - 1) // 2
        self.correlation_projection = nn.Linear(num_pairs * (d_model // 2), d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        batch_size, seq_len, n_joints = x.size()
        pair_embeddings = []
        for i in range(n_joints):
            for j in range(i+1, n_joints):
                joint_pair = torch.cat([x[:, :, i:i+1], x[:, :, j:j+1]], dim=2)
                pair_emb = self.joint_pair_embedding(joint_pair)
                pair_embeddings.append(pair_emb)
        combined_pairs = torch.cat(pair_embeddings, dim=2)
        correlation_features = self.correlation_projection(combined_pairs)
        correlation_features = self.norm(correlation_features)
        correlation_features = self.dropout(correlation_features)
        return correlation_features

class JointTrajectoryTransformer(nn.Module):
    def __init__(self, n_joints=4, d_model=128, n_head=8, n_layers=6, dropout=0.2, n_velocities=4):
        super().__init__()
        self.base_input_dim = n_joints + n_velocities + 3 + 1  # 4 + 4 + 3 + 1 = 12
        # 2의 제곱수로 d_model 유지 (기본값 128은 이미 2^7)
        self.d_model = d_model
        
        # d_model을 더 정밀하게 나누기
        # 예: 128차원을 43+43+42로 나누기
        third = self.d_model // 3
        self.joint_dim = third
        self.velocity_dim = third
        self.endeffector_dim = self.d_model - (2 * third)  # 나머지 차원
        
        # 세 임베딩의 차원 합이 정확히 d_model이 되도록 설정
        self.joint_embedding = nn.Linear(n_joints, self.joint_dim)
        self.velocity_embedding = nn.Linear(n_velocities, self.velocity_dim)
        self.endeffector_embedding = nn.Linear(3, self.endeffector_dim)
        
        # 타임스탬프 임베딩은 전체 d_model 차원으로 출력
        self.timestamp_embedding = nn.Sequential(
            nn.Linear(1, self.d_model // 4),
            nn.GELU(),
            nn.Linear(self.d_model // 4, self.d_model)
        )
        
        # self.joint_embedding = nn.Linear(n_joints, d_model // 3)
        # self.velocity_embedding = nn.Linear(n_velocities, d_model // 3)
        # self.endeffector_embedding = nn.Linear(3, d_model // 3)
        # self.timestamp_embedding = nn.Sequential(
        #     nn.Linear(1, d_model // 4),
        #     nn.GELU(),
        #     nn.Linear(d_model // 4, d_model)
        # )
        self.joint_correlation_module = JointCorrelationModule(n_joints, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length=5000)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.joint_attention_layers = nn.ModuleList([
            MultiHeadJointAttention(d_model, n_head) for _ in range(n_layers // 2)
        ])
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
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
    
    def forward(self, x, velocities, endeffector_positions=None, timestamps=None):
        batch_size, seq_len, _ = x.size()
        joint_features = self.joint_embedding(x)
        velocity_features = self.velocity_embedding(velocities)
        endeffector_features = self.endeffector_embedding(endeffector_positions)
        combined_features = torch.cat([joint_features, velocity_features, endeffector_features], dim=-1)
        if timestamps is not None:
            time_features = self.timestamp_embedding(timestamps)
            combined_features = combined_features + time_features
        correlation_features = self.joint_correlation_module(x)
        combined_features = combined_features + 0.5 * correlation_features
        combined_features = self.positional_encoding(combined_features)
        combined_features = self.input_norm(combined_features)
        combined_features = self.input_dropout(combined_features)
        joint_features = combined_features
        for attention_layer in self.joint_attention_layers:
            joint_attention_output = attention_layer(joint_features)
            joint_features = joint_features + joint_attention_output
        x = self.transformer(joint_features)
        x = self.output_norm(x)
        x = self.output_dropout(x)
        output = self.output_layer(x)
        return output

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        self.data = []
        for traj in trajectories:
            # 데이터프레임인 경우 (전처리된 데이터)
            if isinstance(traj, pd.DataFrame):
                # 필요한 열 추출
                angles = traj[['deg1', 'deg2', 'deg3', 'deg4']].values
                
                # 각속도 - 데이터프레임에 있으면 사용, 없으면 계산
                if all(col in traj.columns for col in ['degsec1', 'degsec2', 'degsec3', 'degsec4']):
                    velocities = traj[['degsec1', 'degsec2', 'degsec3', 'degsec4']].values
                else:
                    velocities = np.gradient(angles, axis=0)
                
                # 엔드이펙터 위치 - 데이터프레임에 있으면 사용, 없으면 계산
                if all(col in traj.columns for col in ['x_end', 'y_end', 'z_end']):
                    endeffector_positions = traj[['x_end', 'y_end', 'z_end']].values
                else:
                    endeffector_positions = np.array([calculate_end_effector_position(deg) for deg in angles])
                
                # 타임스탬프 생성
                if 'time' in traj.columns:
                    time_values = traj['time'].values
                    normalized_time = (time_values - time_values.min()) / (time_values.max() - time_values.min() + 1e-10)
                    timestamps = normalized_time.reshape(-1, 1)
                else:
                    timestamps = np.arange(len(angles)).reshape(-1, 1) / len(angles)
            
            # NumPy 배열인 경우 (기존 방식)
            else:
                angles = traj[:, :4]
                velocities = np.gradient(angles, axis=0)
                endeffector_positions = np.array([calculate_end_effector_position(deg) for deg in angles])
                timestamps = np.arange(len(angles)).reshape(-1, 1) / len(angles)
            
            # 데이터 결합
            combined = np.hstack([angles, velocities, endeffector_positions, timestamps])
            self.data.append(torch.FloatTensor(combined))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class GenerationModel:
    def __init__(self, base_dir=None, model_save_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_dir = base_dir or os.path.join(os.getcwd(), "data")
        if model_save_path:
            self.model_path = model_save_path
        else:
            self.model_path = os.path.join(os.getcwd(), "best_generation_model.pth")
        self.model = JointTrajectoryTransformer().to(self.device)
        try:
            self.analyzer = TrajectoryAnalyzer(
                classification_model="best_classification_model.pth",
                base_dir=self.base_dir
            )
        except Exception as e:
            print(f"Analyzer initialization error: {str(e)}")
            self.analyzer = None
    
    def collect_training_data(self, data_dir=None, n_samples=100):
        """학습 데이터 수집 및 전처리"""
        if self.analyzer is None:
            print("Analyzer not initialized. Cannot collect training data.")
            return []
                
        base_dir = data_dir or os.path.join(self.base_dir, "all_golden_sample")
        trajectories = []
        
        try:
            trajectory_files = [f for f in os.listdir(base_dir) if f.endswith('.txt')]
            if len(trajectory_files) > n_samples:
                trajectory_files = random.sample(trajectory_files, n_samples)
            
            print(f"Collecting data from {len(trajectory_files)} trajectory files...")
            
            for file_name in tqdm(trajectory_files, desc="Loading files"):
                file_path = os.path.join(base_dir, file_name)
                trajectory_type = None
                for type_name in ['d_', 'clock', 'counter', 'v_', 'h_']:
                    if type_name in file_name.lower():
                        trajectory_type = type_name
                        break
                
                if trajectory_type:
                    try:
                        # 파일 내용 읽기
                        with open(file_path, 'r') as f:
                            data_list = []
                            for line in f:
                                parts = line.strip().split(',')
                                if len(parts) >= 7:  # 필요한 모든 열이 있는지 확인
                                    data_list.append(parts)
                        
                        # 전처리 적용
                        preprocessed_df = preprocess_trajectory_data(data_list)
                        
                        # 충분한 데이터 포인트가 있는지 확인
                        if len(preprocessed_df) >= 30:
                            trajectories.append(preprocessed_df)
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
        smoothness_criterion = nn.MSELoss()
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
                angles = batch[:, :, :4]
                velocities = batch[:, :, 4:8]
                endeffector_positions = batch[:, :, 8:11]
                timestamps = batch[:, :, 11:12]
                output_angles = self.model(angles, velocities, endeffector_positions, timestamps)
                angle_loss = criterion(output_angles, angles)
                smoothness_loss = smoothness_criterion(output_angles[:, 1:, :], output_angles[:, :-1, :])
                correlation_loss = 0.0
                for i in range(4):
                    for j in range(i+1, 4):
                        target_corr = angles[:, :, i] - angles[:, :, j]
                        pred_corr = output_angles[:, :, i] - output_angles[:, :, j]
                        correlation_loss += criterion(pred_corr, target_corr)
                predicted_endeffectors = torch.zeros((angles.size(0), angles.size(1), 3), device=self.device)
                for t in range(angles.size(1)):
                    for b in range(angles.size(0)):
                        deg = output_angles[b, t, :].cpu().detach().numpy()
                        pos = calculate_end_effector_position(deg)
                        predicted_endeffectors[b, t, :] = torch.tensor(pos, device=self.device)
                endeffector_loss = criterion(predicted_endeffectors, endeffector_positions)
                total_batch_loss = angle_loss + 0.1 * smoothness_loss + 0.05 * correlation_loss + 0.1 * endeffector_loss
                total_batch_loss.backward()
                optimizer.step()
                total_loss += total_batch_loss.item()
                epoch_progress.set_postfix(loss=f"{total_batch_loss.item():.4f}")
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