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

class TrajectoryDataset(Dataset):
    def __init__(self, base_dir="data", max_seq_length=100, normalize=True):
        self.max_seq_length = max_seq_length
        self.normalize = normalize
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
            target_df, _ = self.analyzer.load_target_trajectory(traj_type, user_df=user_df)
            pairs.append((user_df, target_df))

        return pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_df, target_df = self.samples[idx]

        user_input = self.process_trajectory(user_df)
        target_input = self.process_trajectory(target_df)

        interpolated_gt = (user_input + target_input) / 2.0

        return {
            'user_input': torch.FloatTensor(user_input),
            'target_input': torch.FloatTensor(target_input),
            'interpolated_gt': torch.FloatTensor(interpolated_gt)
        }

    def process_trajectory(self, df):
        degrees = df[['deg1', 'deg2', 'deg3', 'deg4']].values
        endpoints = np.array([calculate_end_effector_position(deg) for deg in degrees]) * 1000
        combined = np.concatenate([endpoints, degrees], axis=1)

        t_orig = np.linspace(0, 1, len(combined))
        t_new = np.linspace(0, 1, self.max_seq_length)

        resampled = np.zeros((self.max_seq_length, combined.shape[1]))
        for i in range(combined.shape[1]):
            spline = UnivariateSpline(t_orig, combined[:, i], s=0)
            resampled[:, i] = spline(t_new)

        if self.normalize:
            mean = np.mean(resampled, axis=0)
            std = np.std(resampled, axis=0) + 1e-8
            resampled = (resampled - mean) / std

        return resampled

class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=7, d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, input_dim)
        self.shape_preservation_weight = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, user_input, target_input, interpolated_gt=None):
        user_emb = self.positional_encoding(self.embedding(user_input))
        target_emb = self.positional_encoding(self.embedding(target_input))

        w = torch.sigmoid(self.shape_preservation_weight)
        memory = w * user_emb + (1 - w) * target_emb

        if interpolated_gt is not None:
            tgt_emb = self.embedding(interpolated_gt[:, :-1, :])
            tgt_emb = self.positional_encoding(tgt_emb)
            tgt_mask = self._generate_square_subsequent_mask(tgt_emb.size(1)).to(tgt_emb.device)
            out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            return self.output_layer(out)
        else:
            B, L, _ = user_input.size()
            generated = torch.zeros(B, L, user_input.size(2)).to(user_input.device)
            for t in range(1, L):
                dec_in = self.embedding(generated[:, :t, :])
                dec_in = self.positional_encoding(dec_in)
                tgt_mask = self._generate_square_subsequent_mask(t).to(user_input.device)
                out = self.decoder(dec_in, memory, tgt_mask=tgt_mask)
                next_val = self.output_layer(out[:, -1:, :])
                generated[:, t, :] = next_val.squeeze(1)
            return generated

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    
    def generate_with_weight(self, user_input, target_input, weight=None):
        """ 지정된 가중치로 궤적 생성 """
        user_emb = self.positional_encoding(self.embedding(user_input))
        target_emb = self.positional_encoding(self.embedding(target_input))
        
        if weight is not None:
            # 외부에서 제공된 가중치 사용
            w = torch.tensor([weight], device=user_input.device)
        else:
            # 학습된 가중치 사용
            w = torch.sigmoid(self.shape_preservation_weight)
        
        # w: 사용자 가중치, (1-w): 타겟 가중치
        memory = w * user_emb + (1 - w) * target_emb
        
        # 자기회귀 생성 과정 구현
        B, L, _ = user_input.size()
        generated = torch.zeros(B, L, user_input.size(2)).to(user_input.device)
        
        for t in range(L):
            if t == 0:
                # 첫 시점은 보간된 값에서 시작 (아니면 사용자/타겟 선택 가능)
                generated[:, 0, :] = w * user_input[:, 0, :] + (1-w) * target_input[:, 0, :]
            else:
                dec_in = self.embedding(generated[:, :t, :])
                dec_in = self.positional_encoding(dec_in)
                tgt_mask = self._generate_square_subsequent_mask(t).to(user_input.device)
                out = self.decoder(dec_in, memory[:, :t, :], tgt_mask=tgt_mask)
                next_val = self.output_layer(out[:, -1:, :])
                generated[:, t, :] = next_val.squeeze(1)
        
        return generated
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, dataloader, device, epochs=100, lr=1e-5, save_path="best_trajectory_transformer.pth", patience=10, seed=42):
    set_seed(seed)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    patience_counter = 0
    print("Training Start")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(dataloader, desc=f"[Epoch {epoch+1}/{epochs}]")

        for batch in progress:
            user_input = batch['user_input'].to(device)
            target_input = batch['target_input'].to(device)
            interpolated_gt = batch['interpolated_gt'].to(device)

            output = model(user_input, target_input, interpolated_gt)
            target = interpolated_gt[:, 1:, :]

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, save_path)
            print(f"Best model saved at epoch {epoch+1}, loss={best_loss:.6f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    base_dir = "data"
    dataset = TrajectoryDataset(base_dir=base_dir, max_seq_length=100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = TrajectoryTransformer()
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
