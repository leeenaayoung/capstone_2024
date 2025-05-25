import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import math
import random
import os
import numpy as np
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import UnivariateSpline, CubicSpline
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from tqdm import tqdm
from utils import *
from analyzer import TrajectoryAnalyzer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

    # def __init__(self, d_model, max_len=5000):
    #     super().__init__()
    #     self.encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
    #     nn.init.normal_(self.encoding, mean=0.0, std=0.02)

    # def forward(self, x):
    #     return x + self.encoding[:, :x.size(1), :]

class TrajectoryDataset(Dataset):
    def __init__(self, base_dir="data", max_seq_length=100):
        self.max_seq_length = max_seq_length
        self.analyzer = TrajectoryAnalyzer(
            classification_model="best_classification_model.pth",
            base_dir=base_dir
        )
        self.samples = self.prepare_interpolation_pairs(
            non_golden_dir=os.path.join(base_dir, "all_golden_sample")
        )

    def prepare_interpolation_pairs(self, non_golden_dir):
        pairs = []
        user_files = [f for f in os.listdir(non_golden_dir) if f.endswith('.txt')]
        for file in user_files:
            user_path = os.path.join(non_golden_dir, file)
            user_df, traj_type = self.analyzer.load_user_trajectory(user_path)
            target_df, target_file = self.analyzer.load_target_trajectory(traj_type, user_df=user_df)
            pairs.append((user_df, target_df))
        return pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_df, target_df = self.samples[idx]
        user_input, user_mean, user_std = self.process_trajectory(user_df)
        target_input, target_mean, target_std = self.process_trajectory(target_df)

        alpha = np.random.uniform(0.2, 0.8)
        aligned_user, aligned_target = self.apply_dtw_alignment(user_df, target_df)

        aligned_user_input, _, _ = self.process_trajectory(aligned_user)
        aligned_target_input, _, _ = self.process_trajectory(aligned_target)

        interpolated_gt = alpha * aligned_user_input + (1 - alpha) * aligned_target_input
        # interpolated_gt = self.apply_temporal_smoothing(interpolated_gt)

        return {
            'user_input': torch.FloatTensor(user_input),
            'target_input': torch.FloatTensor(target_input),
            'interpolated_gt': torch.FloatTensor(interpolated_gt),
            'alpha': torch.FloatTensor([alpha]),
            'user_mean': torch.FloatTensor(user_mean),
            'user_std': torch.FloatTensor(user_std)
        }
    
    def apply_dtw_alignment(self, user_df, target_df):
        user_deg = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        target_deg = target_df[['deg1', 'deg2', 'deg3', 'deg4']].values

        user_ee = np.array([calculate_end_effector_position(d) for d in user_deg]) * 1000
        target_ee = np.array([calculate_end_effector_position(d) for d in target_deg]) * 1000

        _, path = fastdtw(user_ee, target_ee, dist=euclidean)
        path = np.array(path)

        aligned_user = user_df.iloc[path[:, 0]].reset_index(drop=True)
        aligned_target = target_df.iloc[path[:, 1]].reset_index(drop=True)

        return aligned_user, aligned_target

    # def apply_temporal_smoothing(self, trajectory, smoothing_factor=0.1):
    #     smoothed = trajectory.copy()
    #     window_size = max(3, int(len(trajectory) * smoothing_factor))
    #     if window_size % 2 == 0:
    #         window_size += 1
    #     for dim in range(trajectory.shape[1]):
    #         for i in range(len(trajectory)):
    #             start_idx = max(0, i - window_size // 2)
    #             end_idx = min(len(trajectory), i + window_size // 2 + 1)
    #             smoothed[i, dim] = np.mean(trajectory.iloc[start_idx:end_idx, dim])
    #     return smoothed
    
    def process_trajectory(self, df):
        degrees = df[['deg1', 'deg2', 'deg3', 'deg4']].values
        endpoints = np.array([calculate_end_effector_position(deg) for deg in degrees]) * 1000
        combined = np.concatenate([endpoints, degrees], axis=1)
        t_orig = np.linspace(0, 1, len(combined))
        t_new = np.linspace(0, 1, self.max_seq_length)
        resampled = np.zeros((self.max_seq_length, combined.shape[1]))
        for i in range(combined.shape[1]):
            spline = CubicSpline(t_orig, combined[:, i])
            resampled[:, i] = spline(t_new)
        mean = np.mean(resampled, axis=0)
        std = np.std(resampled, axis=0) + 1e-8
        normalized = (resampled - mean) / std
        return normalized, mean, std

class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=7, d_model=64, nhead=2, num_layers=2, dropout=0.05):
        super().__init__()
        self.input_dim = input_dim
        self.user_embedding = nn.Linear(input_dim, d_model)
        self.target_embedding = nn.Linear(input_dim, d_model)
        self.output_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.user_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2
        )
        self.target_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2
        )

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, input_dim)

        self.weight_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 1), nn.Sigmoid()
        )

    def forward(self, user_input, target_input, interpolated_gt=None, training_alpha=None):
        user_emb = self.positional_encoding(self.user_embedding(user_input))
        target_emb = self.positional_encoding(self.target_embedding(target_input))

        user_encoded = self.user_encoder(user_emb)
        target_encoded = self.target_encoder(target_emb)

        context = torch.cat([torch.mean(user_encoded, dim=1), torch.mean(target_encoded, dim=1)], dim=1)
        adaptive_weight = self.weight_predictor(context).unsqueeze(1)
        weight = training_alpha.unsqueeze(-1) if training_alpha is not None and self.training else adaptive_weight

        memory = weight * user_encoded + (1 - weight) * target_encoded

        if interpolated_gt is not None:
            return self.forward_teacher_forcing(interpolated_gt, memory), adaptive_weight.squeeze()
        else:
            return self.generate_autoregressive(user_input, target_input, memory, weight)

    def forward_teacher_forcing(self, interpolated_gt, memory):
        tgt_emb = self.positional_encoding(self.output_embedding(interpolated_gt[:, :-1, :]))
        tgt_mask = self._generate_square_subsequent_mask(tgt_emb.size(1)).to(tgt_emb.device)
        decoder_output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.output_layer(decoder_output)

    def generate_autoregressive(self, user_input, target_input, memory, weight):
        batch_size, seq_len, input_dim = user_input.size()
        generated = torch.zeros(batch_size, seq_len, input_dim).to(user_input.device)
        generated[:, 0, :] = weight.squeeze(-1) * user_input[:, 0, :] + (1 - weight.squeeze(-1)) * target_input[:, 0, :]
        for t in range(1, seq_len):
            current_input = generated[:, :t, :]
            tgt_emb = self.positional_encoding(self.output_embedding(current_input))
            tgt_mask = self._generate_square_subsequent_mask(t).to(user_input.device)
            decoder_output = self.decoder(tgt_emb, memory[:, :t, :], tgt_mask=tgt_mask)
            next_step = self.output_layer(decoder_output[:, -1:, :])
            generated[:, t, :] = next_step.squeeze(1)
        return generated, weight.squeeze()

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz)) == 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, dataloader, device, epochs=100, lr=1e-4, save_path="best_trajectory_transformer.pth", patience=10, seed=42):
    set_seed(seed)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    mse_criterion = nn.MSELoss()
    smooth_criterion = nn.MSELoss()
    
    best_loss = float('inf')
    patience_counter = 0 
    
    print("Training Start")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_mse_loss = 0.0
        total_smooth_loss = 0.0
        
        progress = tqdm(dataloader, desc=f"[Epoch {epoch+1}/{epochs}]")

        for batch in progress:
            user_input = batch['user_input'].to(device)
            target_input = batch['target_input'].to(device)
            interpolated_gt = batch['interpolated_gt'].to(device)
            alpha = batch['alpha'].to(device)

            # 모델 forward
            output, predicted_weight = model(
                user_input, target_input, 
                interpolated_gt=interpolated_gt, 
                training_alpha=alpha
            )
            
            target_sequence = interpolated_gt[:, 1:, :]

            # 주요 손실: MSE 손실
            mse_loss = mse_criterion(output, target_sequence)
            
            # 보조 손실 1: 시간적 스무딩 손실 (급격한 변화 방지)
            if output.size(1) > 1:
                diff_pred = output[:, 1:, :] - output[:, :-1, :]
                diff_target = target_sequence[:, 1:, :] - target_sequence[:, :-1, :]
                smooth_loss = smooth_criterion(diff_pred, diff_target)
            else:
                smooth_loss = torch.tensor(0.0).to(device)
            
            # 보조 손실 2: 가중치 예측 손실
            weight_loss = mse_criterion(predicted_weight, alpha.squeeze())
            
            # 전체 손실 결합
            total_batch_loss = mse_loss + 0.1 * smooth_loss + 0.05 * weight_loss

            optimizer.zero_grad()
            total_batch_loss.backward()
            
            # 그래디언트 클리핑 (안정성 향상)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += total_batch_loss.item()
            total_mse_loss += mse_loss.item()
            total_smooth_loss += smooth_loss.item()
            
            progress.set_postfix({
                'total': total_batch_loss.item(),
                'mse': mse_loss.item(),
                'smooth': smooth_loss.item()
            })

        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse_loss / len(dataloader)
        avg_smooth = total_smooth_loss / len(dataloader)
        
        print(f"Epoch {epoch+1} - Total: {avg_loss:.6f}, MSE: {avg_mse:.6f}, Smooth: {avg_smooth:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, save_path)
            print(f"최고 모델 저장: epoch {epoch+1}, loss={best_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"개선 없음. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("조기 종료 실행")
            break

def main():
    base_dir = "data"
    dataset = TrajectoryDataset(base_dir=base_dir, max_seq_length=100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = TrajectoryTransformer(
                                    input_dim=7, 
                                    d_model=64, 
                                    nhead=2, 
                                    num_layers=2
                                )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(
        model, 
        dataloader, 
        device, 
        epochs=100, 
        lr=1e-4,
        save_path="best_trajectory_transformer.pth",
        patience=10,       
        seed=42            
    )

if __name__ == "__main__":
    main()
