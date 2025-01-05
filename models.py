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
# print(f"Using device: {device}")
scaler = StandardScaler()

# 가중치 초기화
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

##########################
# 분류 데이터셋 로드
##########################
class ClassificationDataset(Dataset):
    def __init__(self, base_path):
        self.data_cache = {}
        self.data = []
        self.labels = []
        self.load_data(base_path)

        # 모든 데이터를 모은 후, 스케일링 적용
        all_data = torch.cat(self.data, dim=0)  # (row-wise) 통합
        scaler.fit(all_data)  # 스케일러에 fit

        # 스케일링 후 다시 저장
        scaled_data_list = []
        for sample in self.data:
            sample_np = sample.numpy()
            sample_scaled = scaler.transform(sample_np)
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
def collate_fn(batch):
    data_list = [item[0] for item in batch]
    label_list = [item[1] for item in batch]

    data_padded = pad_sequence(data_list, batch_first=True, padding_value=0)
    label_indices = [unique_labels.index(lbl) for lbl in label_list]
    label_indices = torch.tensor(label_indices, dtype=torch.long)

    return data_padded, label_indices

def collate_fn(batch, base_path):
    # batch: list of (data_tensor, label_str)
    
    # unique_labels 얻기
    unique_labels = get_unique_labels(base_path)  # 여기서 unique_labels를 얻음
    
    data_list = [item[0] for item in batch]
    label_list = [item[1] for item in batch]

    # pad_sequence -> (batch_size, max_len, feature_dim)
    data_padded = pad_sequence(data_list, batch_first=True, padding_value=0)

    # 라벨 인덱스 변환
    label_indices = [unique_labels.index(lbl) for lbl in label_list]
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
    
##########################
# 생성 데이터셋 로드
##########################
class GenerationTrajectoryLoader:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.golden_dir = os.path.join(base_dir, "golden_sample")
        self.non_golden_dir = os.path.join(base_dir, "non_golden_sample")

    def get_random_trajectory(self, folder_path, movement_type):
        # 파일 선택 및 읽기 로직
        trajectory_files = [f for f in os.listdir(folder_path) 
                          if f.endswith('.txt') and f.startswith(movement_type)]
        
        if not trajectory_files:
            return None, None
            
        selected_file = random.choice(trajectory_files)
        file_path = os.path.join(folder_path, selected_file)
        
        try:
            with open(file_path, 'r') as f:
                data_list = list(csv.reader(f))
            return data_list, selected_file
        except Exception as e:
            print(f"Error loading {selected_file}: {str(e)}")
            return None, None

    def collect_trajectories(self, movement_type, num_samples=5):
        # Golden 샘플 수집
        trajectories = []
        for _ in range(num_samples):
            data_list, file_name = self.get_random_trajectory(self.golden_dir, movement_type)
            if data_list is not None:
                trajectories.append(data_list)
        return trajectories
    
class GenerationDataset(Dataset):
    def __init__(self, data_list, sequence_length=50):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 데이터 전처리
        data_v = preprocess_trajectory_data(data_list)
        print("Preprocessed data shape:", data_v.shape)
        
        # 필요한 특성만 선택
        self.features = ['x_end', 'y_end', 'z_end', 'yaw', 'pitch', 'roll',
                        'deg1', 'deg2', 'deg3', 'deg4',
                        'torque1', 'torque2', 'torque3', 'force1', 'force2', 'force3']
        data = data_v[self.features].values
        print("Selected features shape:", data.shape)
        
        # 데이터 스케일링
        self.scaler = MinMaxScaler(feature_range=(-1, 1)) 
        scaled_data = self.scaler.fit_transform(data)
        print("Scaled data shape:", scaled_data.shape)
        
        # 시퀀스 생성
        self.sequences = []
        self.targets = []
        for i in range(len(scaled_data) - sequence_length):
            self.sequences.append(scaled_data[i:i + sequence_length])
            self.targets.append(scaled_data[i + sequence_length])
        
        # numpy 배열로 변환
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
        
        # 텐서로 변환 및 GPU로 이동
        self.sequences = torch.FloatTensor(self.sequences)
        self.targets = torch.FloatTensor(self.targets)
        
        print(f"Number of sequences: {len(self.sequences)}")
        print(f"Sequence shape: {self.sequences.shape}")
        print(f"Target shape: {self.targets.shape}")
            
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.sequences[idx]), 
                torch.FloatTensor(self.targets[idx]))

##########################
# 생성 모델 정의
##########################
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=False  # 단방향 학습
        )

        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output
