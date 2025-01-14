import os
import csv
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch import optim
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from utils import preprocess_trajectory_data

##########################
# 기본 설정 및 시드 고정
##########################
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = StandardScaler()
global unique_labels

# 가중치 초기화
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

##########################
# 데이터셋 로드
##########################
class ClassificationDataset(Dataset):
    def __init__(self, base_path):
        self.data_cache = {}
        self.data = []
        self.labels = []
        self.scaler = StandardScaler()
        self.load_data(base_path)
        self.unique_labels = sorted(list(set(self.labels))) 

        # 모든 데이터를 모은 후, 스케일링 적용
        all_data = torch.cat(self.data, dim=0)  # (row-wise) 통합
        all_data_np = all_data.numpy()
        self.scaler.fit(all_data_np)  # 스케일러에 fit

        # 스케일링 후 다시 저장
        scaled_data_list = []
        for sample in self.data:
            sample_np = sample.numpy()
            sample_scaled = self.scaler.transform(sample_np)
            scaled_data_list.append(torch.tensor(sample_scaled, dtype=torch.float32))

        self.data = scaled_data_list  # 최종 교체

    def load_data(self, base_path):
        """txt 파일명에서 두 번째 언더바 전까지를 라벨로 사용"""
        for file_name in sorted(os.listdir(base_path)):
            if file_name.endswith('.txt'):
                # 파일명에서 라벨 추출 (두 번째 언더바 전까지)
                label = '_'.join(file_name.split('_')[:2])  # 첫 번째와 두 번째 부분만 가져옴
                
                file_path = os.path.join(base_path, file_name)
                with open(file_path, 'r') as f:
                    data_list = list(csv.reader(f))
                
                # 데이터 전처리
                data_v = preprocess_trajectory_data(data_list)
                tensor_data = torch.tensor(data_v.values, dtype=torch.float32)

                self.data.append(tensor_data)
                self.labels.append(label)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
# 라벨 목록
def get_unique_labels(base_path):
    tmp_dataset = ClassificationDataset(base_path)
    return sorted(list(set(tmp_dataset.labels)))

# 기존
def collate_fn(batch, dataset):
    global unique_labels  
    data_list = [item[0] for item in batch]
    label_list = [item[1] for item in batch]

    data_padded = pad_sequence(data_list, batch_first=True, padding_value=0)
    label_indices = [dataset.unique_labels.index(lbl) for lbl in label_list]
    label_indices = torch.tensor(label_indices, dtype=torch.long)

    return data_padded, label_indices

##########################
# 분류 모델 정의
##########################
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes, max_len=2000):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, d_model))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        seq_len = x.size(1)
        pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = self.embedding(x) + pos_enc
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        out = self.fc(x)
        return out
    