import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import math
import random
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import calculate_end_effector_position
from analyzer import TrajectoryAnalyzer

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TrajectoryDataset(Dataset):
    """궤적 데이터셋 클래스 (시간적, 공간적 특성 고려)"""
    def __init__(self, base_dir="data", max_seq_length=100, normalize=True):
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.analyzer = TrajectoryAnalyzer(classification_model="best_classification_model.pth", 
                                          base_dir=base_dir)
        
        # 데이터 수집
        self.trajectories, self.trajectory_types, self.spatial_encodings = self.collect_data(
            golden_dir=os.path.join(base_dir, "golden_sample"),
            non_golden_dir=os.path.join(base_dir, "non_golden_sample")
        )
        
        # 타입을 숫자로 인코딩
        self.type_to_idx = {
            'd_': 0,      # 대각선
            'clock': 1,   # 시계 방향
            'counter': 2, # 반시계 방향
            'v_': 3,      # 수직
            'h_': 4       # 수평
        }

    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        # 궤적 데이터와 타입 가져오기
        trajectory = self.trajectories[idx]
        traj_type = self.trajectory_types[idx]
        spatial_encoding = self.spatial_encodings[idx]
        
        # 각도 데이터
        angles = trajectory[['deg1', 'deg2', 'deg3', 'deg4']].values
        
        # 엔드이펙터 위치 계산
        endpoints = np.array([calculate_end_effector_position(deg) for deg in angles]) * 1000
        
        # 시퀀스 길이 조정
        seq_length = len(endpoints)
        
        # 타임스탬프 정규화 (0~1 범위)
        if 'timestamp' in trajectory.columns:
            timestamps = trajectory['timestamp'].values
            timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
        else:
            # 타임스탬프가 없으면 균등 간격 사용
            timestamps = np.linspace(0, 1, seq_length)
        
        # 시퀀스 길이가 max_seq_length보다 길면 균등하게 샘플링
        if seq_length > self.max_seq_length:
            indices = np.linspace(0, seq_length-1, self.max_seq_length).astype(int)
            endpoints = endpoints[indices]
            angles = angles[indices]
            timestamps = timestamps[indices]
            seq_length = self.max_seq_length
        
        # 데이터 정규화 (선택적)
        if self.normalize:
            # 공간적 위치 정규화 (단위 변환 및 중심 이동)
            endpoints_mean = np.mean(endpoints, axis=0)
            endpoints_std = np.std(endpoints, axis=0)
            normalized_endpoints = (endpoints - endpoints_mean) / (endpoints_std + 1e-8)
            
            # 각도 정규화 (-1~1 범위)
            angles_normalized = angles / 180.0  # 각도 범위를 -1~1로 조정
            
            # 정규화된 값 사용
            endpoints = normalized_endpoints
            angles = angles_normalized
        
        # 타입 원-핫 인코딩
        type_idx = self.type_to_idx.get(traj_type, 0)
        type_onehot = np.zeros(len(self.type_to_idx))
        type_onehot[type_idx] = 1
        
        # 여기서 모든 배열의 길이가 일치하는지 확인
        assert len(timestamps) == len(endpoints) == len(angles), "배열 길이 불일치"
        
        # 모든 특성 결합
        # [타임스탬프, 위치(3), 각도(4), 타입(5), 공간인코딩(3)]
        combined_features = np.column_stack([
            # timestamps.reshape(-1, 1),              # 시간 정보
            endpoints,                              # 3D 위치
            angles,                                 # 관절 각도
        ])
        
        # 패딩 (시퀀스 길이가 max_seq_length보다 작은 경우)
        if seq_length < self.max_seq_length:
            padding = np.zeros((self.max_seq_length - seq_length, combined_features.shape[1]))
            combined_features = np.vstack([combined_features, padding])
            
            # 패딩된 부분을 마스킹하기 위한 어텐션 마스크 생성
            attention_mask = np.ones(self.max_seq_length)
            attention_mask[seq_length:] = 0
        else:
            attention_mask = np.ones(self.max_seq_length)
        
        return {
            'features': torch.FloatTensor(combined_features),
            'attention_mask': torch.FloatTensor(attention_mask),
            'seq_length': seq_length,
            'traj_type': traj_type,
            'spatial_encoding': torch.FloatTensor(spatial_encoding)
        }
    
    def collect_data(self, golden_dir, non_golden_dir, n_samples=None):
        """데이터 수집 (표준 + 사용자 궤적)"""
        trajectories = []
        trajectory_types = []
        spatial_encodings = []
        
        # 표준 궤적 (golden samples) 로드
        golden_files = [f for f in os.listdir(golden_dir) if f.endswith('.txt')]
        
        # 사용자 궤적 (non-golden samples) 로드
        non_golden_files = [f for f in os.listdir(non_golden_dir) if f.endswith('.txt')]
        
        # 샘플 수 제한 (선택적)
        if n_samples is not None:
            if len(golden_files) > n_samples // 2:
                golden_files = random.sample(golden_files, n_samples // 2)
            if len(non_golden_files) > n_samples // 2:
                non_golden_files = random.sample(non_golden_files, n_samples // 2)
        
        # 표준 궤적 처리
        for filename in golden_files:
            trajectory_type = None
            for type_name in ['d_', 'clock', 'counter', 'v_', 'h_']:
                if type_name in filename.lower():
                    trajectory_type = type_name
                    break
            
            if trajectory_type:
                try:
                    file_path = os.path.join(golden_dir, filename)
                    df, _ = self.analyzer.load_user_trajectory(file_path)
                    
                    # 공간적 위치 인코딩 (궤적의 시작점, 끝점, 중심점으로 공간 정보 표현)
                    spatial_encoding = self.compute_spatial_encoding(df)
                    
                    trajectories.append(df)
                    trajectory_types.append(trajectory_type)
                    spatial_encodings.append(spatial_encoding)
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        # 사용자 궤적 처리
        for filename in non_golden_files:
            try:
                file_path = os.path.join(non_golden_dir, filename)
                df, trajectory_type = self.analyzer.load_user_trajectory(file_path)
                
                # 공간적 위치 인코딩
                spatial_encoding = self.compute_spatial_encoding(df)
                
                trajectories.append(df)
                trajectory_types.append(trajectory_type)
                spatial_encodings.append(spatial_encoding)
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
        
        return trajectories, trajectory_types, spatial_encodings
    
    def compute_spatial_encoding(self, trajectory_df):
        """궤적의 공간적 위치 인코딩 (시작점, 끝점, 중심점 사용)"""
        angles = trajectory_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        endpoints = np.array([calculate_end_effector_position(deg) for deg in angles]) * 1000
        
        # 시작점, 끝점, 중심점
        start_point = endpoints[0]
        end_point = endpoints[-1]
        center_point = np.mean(endpoints, axis=0)
        
        # 정규화된 특성 벡터 (시작-끝 벡터의 방향, 시작-중심 벡터의 방향, 공간 크기)
        start_end_vector = end_point - start_point
        start_end_dist = np.linalg.norm(start_end_vector)
        if start_end_dist > 0:
            start_end_vector = start_end_vector / start_end_dist
        
        start_center_vector = center_point - start_point
        start_center_dist = np.linalg.norm(start_center_vector)
        if start_center_dist > 0:
            start_center_vector = start_center_vector / start_center_dist
        
        # 궤적의 공간적 크기 (바운딩 박스 대각선 길이)
        spatial_size = np.linalg.norm(np.max(endpoints, axis=0) - np.min(endpoints, axis=0))
        
        # 공간 인코딩 (9차원)
        spatial_encoding = np.concatenate([
            start_point,           # 시작점 (3)
            end_point,             # 끝점 (3)
            center_point,          # 중심점 (3)
            start_end_vector,      # 방향 벡터 (3)
            start_center_vector,   # 방향 벡터 (3)
            [spatial_size]         # 크기 (1)
        ])
        
        return spatial_encoding
    
class PositionalEncoding(nn.Module):
    """시퀀스 위치 인코딩"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TrajectoryTransformer(nn.Module):
    """시간적, 공간적 특성을 모두 고려하는 트랜스포머 모델"""
    def __init__(self, input_dim, d_model=256, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1, 
                 max_seq_length=100, num_trajectory_types=5):
        super(TrajectoryTransformer, self).__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # 특성 임베딩 (입력 특성 → 모델 차원)
        self.feature_embedding = nn.Linear(input_dim, d_model)
        
        # 시퀀스 위치 인코딩
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # 공간 인코딩 처리 (16차원 → d_model)
        self.spatial_embedding = nn.Sequential(
            nn.Linear(16, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 궤적 타입 임베딩
        self.type_embedding = nn.Embedding(num_trajectory_types, d_model)
        
        # 트랜스포머 인코더
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_encoder_layers
        )
        
        # 트랜스포머 디코더
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layers, 
            num_decoder_layers
        )
        
        # 출력 레이어 (위치 예측용)
        self.output_layer = nn.Linear(d_model, 3)  # 3D 위치 예측
        
        # 추가 출력 레이어 (각도 예측용)
        self.angle_output_layer = nn.Linear(d_model, 4)  # 4개 관절 각도 예측
        
    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, 
                src_padding_mask=None, tgt_padding_mask=None, 
                memory_mask=None, memory_key_padding_mask=None,
                spatial_encoding=None, trajectory_type=None):
        """
        src: 입력 시퀀스 [batch_size, seq_len, features]
        tgt: 타겟 시퀀스 (학습 시) [batch_size, tgt_len, features]
        """
        batch_size = src.size(0)
        
        # 입력 시퀀스 임베딩
        src_embedded = self.feature_embedding(src)
        
        # 위치 인코딩 적용
        src_embedded = self.pos_encoder(src_embedded)
        
        # 공간 인코딩 통합 (모든 시퀀스 포인트에 동일하게 적용)
        if spatial_encoding is not None:
            spatial_emb = self.spatial_embedding(spatial_encoding)
            spatial_emb = spatial_emb.unsqueeze(1).expand(-1, src_embedded.size(1), -1)
            src_embedded = src_embedded + spatial_emb
        
        # 궤적 타입 임베딩 통합 (모든 시퀀스 포인트에 동일하게 적용)
        if trajectory_type is not None:
            type_emb = self.type_embedding(trajectory_type)
            type_emb = type_emb.unsqueeze(1).expand(-1, src_embedded.size(1), -1)
            src_embedded = src_embedded + type_emb
        
        # 인코더 통과
        memory = self.transformer_encoder(
            src_embedded, 
            mask=src_mask, 
            src_key_padding_mask=src_padding_mask
        )
        
        if tgt is None:  # 추론 모드
            return memory
        
        # 타겟 시퀀스 임베딩
        tgt_embedded = self.feature_embedding(tgt)
        tgt_embedded = self.pos_encoder(tgt_embedded)
        
        # 마찬가지로 공간 및 타입 인코딩 통합
        if spatial_encoding is not None:
            tgt_embedded = tgt_embedded + spatial_emb[:, :tgt_embedded.size(1), :]
        
        if trajectory_type is not None:
            tgt_embedded = tgt_embedded + type_emb[:, :tgt_embedded.size(1), :]
        
        # 디코더 통과
        output = self.transformer_decoder(
            tgt_embedded, 
            memory, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # 위치 예측
        position_output = self.output_layer(output)
        
        # 각도 예측
        angle_output = self.angle_output_layer(output)
        
        return {
            'positions': position_output,
            'angles': angle_output,
            'hidden_states': output
        }
    
    def generate_trajectory(self, input_seq, trajectory_type_idx, spatial_encoding,
                           max_length=None, temperature=1.0):
        """입력 시퀀스로부터 새로운 궤적 생성"""
        if max_length is None:
            max_length = self.max_seq_length
        
        batch_size = input_seq.size(0)
        device = input_seq.device
        
        # 궤적 타입과 공간 인코딩을 텐서로 변환
        if isinstance(trajectory_type_idx, int):
            trajectory_type = torch.tensor([trajectory_type_idx], device=device)
        else:
            trajectory_type = trajectory_type_idx
            
        if not isinstance(spatial_encoding, torch.Tensor):
            spatial_encoding = torch.tensor(spatial_encoding, dtype=torch.float, device=device)
        
        if spatial_encoding.dim() == 1:
            spatial_encoding = spatial_encoding.unsqueeze(0)
        
        # 인코더 출력 계산
        memory = self.forward(
            src=input_seq,
            spatial_encoding=spatial_encoding,
            trajectory_type=trajectory_type
        )
        
        # 초기 출력 시퀀스 (입력의 마지막 포인트)
        output_seq = input_seq[:, -1:, :3].clone()  # 위치 좌표만 사용
        
        # 자기회귀 생성
        for i in range(max_length - 1):
            # 현재까지의 출력으로 다음 포인트 예측
            curr_input = torch.cat([
                output_seq,
                torch.zeros(batch_size, output_seq.size(1), input_seq.size(2) - 3, device=device)
            ], dim=2)
            
            # 위치 예측
            with torch.no_grad():
                # 마스크 생성 (자기회귀용)
                tgt_mask = self.generate_square_subsequent_mask(curr_input.size(1)).to(device)
                
                # 다음 포인트 예측
                next_output = self.forward(
                    src=input_seq,
                    tgt=curr_input,
                    tgt_mask=tgt_mask,
                    spatial_encoding=spatial_encoding,
                    trajectory_type=trajectory_type
                )
            
            # 마지막 위치 예측값 추출
            next_point = next_output['positions'][:, -1:, :]
            
            # 약간의 무작위성 추가 (optional)
            if temperature > 0:
                noise = torch.randn_like(next_point) * temperature
                next_point = next_point + noise
            
            # 출력 시퀀스에 추가
            output_seq = torch.cat([output_seq, next_point], dim=1)
        
        return output_seq
    
    def generate_square_subsequent_mask(self, sz):
        """디코더의 자기회귀 attention을 위한 마스크 생성"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def interpolate_trajectories(self, trajectory1, trajectory2, interpolate_weight=0.5,
                               traj_type=None, input_length=25, output_length=100):
        """두 궤적 사이의 보간된 새 궤적 생성"""
        device = next(self.parameters()).device
        
        # 각 궤적의 시작 부분 가져오기
        start1 = trajectory1[:input_length]
        start2 = trajectory2[:input_length]
        
        # 시작점 보간
        interpolated_start = (1 - interpolate_weight) * start1 + interpolate_weight * start2
        
        # 공간 인코딩 계산
        if hasattr(self, 'compute_spatial_encoding'):
            # 두 궤적의 공간 인코딩 보간
            spatial_encoding1 = self.compute_spatial_encoding(trajectory1)
            spatial_encoding2 = self.compute_spatial_encoding(trajectory2)
            spatial_encoding = (1 - interpolate_weight) * spatial_encoding1 + interpolate_weight * spatial_encoding2
        else:
            # 간단한 공간 인코딩 (시작, 끝, 중심)
            def simple_spatial_encoding(trajectory):
                start = trajectory[0, :3]
                end = trajectory[-1, :3]
                center = torch.mean(trajectory[:, :3], dim=0)
                return torch.cat([start, end, center])
            
            spatial_encoding1 = simple_spatial_encoding(trajectory1)
            spatial_encoding2 = simple_spatial_encoding(trajectory2)
            spatial_encoding = (1 - interpolate_weight) * spatial_encoding1 + interpolate_weight * spatial_encoding2
        
        # 궤적 타입 결정
        if traj_type is None:
            # 가중치에 따라 타입 결정 (더 높은 가중치를 가진 궤적의 타입 사용)
            traj_type = 2 if interpolate_weight > 0.5 else 1
        
        # 보간된 시작점으로 모델 입력 생성
        interpolated_input = interpolated_start.unsqueeze(0).to(device)
        spatial_encoding = spatial_encoding.unsqueeze(0).to(device)
        
        # 새로운 궤적 생성
        with torch.no_grad():
            generated_trajectory = self.generate_trajectory(
                input_seq=interpolated_input,
                trajectory_type_idx=traj_type,
                spatial_encoding=spatial_encoding,
                max_length=output_length
            )
        
        return generated_trajectory.squeeze(0)
    

def train_model(model, train_loader, val_loader, device, epochs=200, lr=0.0001, save_path="trained_model.pth"):
    """트랜스포머 모델을 학습시키는 함수"""
    import warnings
    warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask and attn_mask is deprecated.")
    # 최적화기 및 손실 함수 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    position_criterion = nn.MSELoss()
    angle_criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    # 에포크 진행 상황을 보여주는 tqdm
    epoch_progress = tqdm(range(epochs), desc="Training Epochs", position=0)
    
    for epoch in epoch_progress:
        # 학습 모드 설정
        model.train()
        train_position_loss = 0
        train_angle_loss = 0
        
        # 배치 진행 상황을 보여주는 tqdm
        batch_progress = tqdm(enumerate(train_loader), 
                             total=len(train_loader), 
                             desc=f"Epoch {epoch+1}/{epochs} (Train)", 
                             leave=False, 
                             position=1)
        
        for i, batch in batch_progress:
            optimizer.zero_grad()
            
            # 데이터 준비
            features = batch['features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            spatial_encoding = batch['spatial_encoding'].to(device)
            
            # 궤적 타입 인덱스 변환
            traj_type_names = batch['traj_type']
            traj_types = torch.tensor([train_loader.dataset.dataset.type_to_idx.get(t, 0) for t in traj_type_names], device=device)
            
            # 패딩 마스크 생성
            src_padding_mask = (attention_mask == 0).to(device)
            
            # 자기회귀 학습을 위한 입력/타겟 분리
            src = features[:, :-1, :]  # 마지막 지점 제외한 모든 포인트
            tgt = features[:, 1:, :]   # 첫 지점 제외한 모든 포인트
            
            # 패딩 마스크 조정
            if src_padding_mask.size(1) > src.size(1):
                src_padding_mask = src_padding_mask[:, :src.size(1)]
            
            tgt_padding_mask = src_padding_mask.clone()
            if tgt_padding_mask.size(1) > tgt.size(1):
                tgt_padding_mask = tgt_padding_mask[:, :tgt.size(1)]
            
            # 자기회귀 마스크 생성
            tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            # 모델 출력
            outputs = model(
                src=src,
                tgt=tgt,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
                spatial_encoding=spatial_encoding,
                trajectory_type=traj_types
            )
            
            # 손실 계산 (위치 및 각도)
            position_loss = position_criterion(outputs['positions'], tgt[:, :, 0:3])
            angle_loss = angle_criterion(outputs['angles'], tgt[:, :, 3:7])
            # 전체 손실
            total_loss = position_loss + 0.5 * angle_loss
            
            # 역전파
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_position_loss += position_loss.item()
            train_angle_loss += angle_loss.item()
            
            # 현재 배치의 손실을 tqdm 진행 막대에 표시
            batch_progress.set_postfix({
                'pos_loss': f'{position_loss.item():.4f}', 
                'angle_loss': f'{angle_loss.item():.4f}'
            })
        
        # 검증 시작
        model.eval()
        val_position_loss = 0
        val_angle_loss = 0
        
        # 검증 데이터에 대한 tqdm 진행 막대
        val_progress = tqdm(val_loader, 
                           desc=f"Epoch {epoch+1}/{epochs} (Validation)", 
                           leave=False, 
                           position=1)
        
        with torch.no_grad():
            for batch in val_progress:
                # 데이터 준비
                features = batch['features'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                spatial_encoding = batch['spatial_encoding'].to(device)
                
                # 궤적 타입 인덱스 변환
                traj_type_names = batch['traj_type']
                traj_types = torch.tensor([val_loader.dataset.dataset.type_to_idx.get(t, 0) for t in traj_type_names], device=device)
                
                # 자기회귀 학습을 위한 입력/타겟 분리
                src = features[:, :-1, :]
                tgt = features[:, 1:, :]
                
                # 패딩 마스크 생성
                src_padding_mask = (attention_mask == 0).to(device)
                if src_padding_mask.size(1) > src.size(1):
                    src_padding_mask = src_padding_mask[:, :src.size(1)]
                
                tgt_padding_mask = src_padding_mask.clone()
                if tgt_padding_mask.size(1) > tgt.size(1):
                    tgt_padding_mask = tgt_padding_mask[:, :tgt.size(1)]
                
                # 자기회귀 마스크 생성
                tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
                
                # 모델 출력
                outputs = model(
                    src=src,
                    tgt=tgt,
                    tgt_mask=tgt_mask,
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask,
                    spatial_encoding=spatial_encoding,
                    trajectory_type=traj_types
                )
                
                # 손실 계산
                position_loss = position_criterion(outputs['positions'], tgt[:, :, 0:3])
                angle_loss = angle_criterion(outputs['angles'], tgt[:, :, 3:7])
                
                val_position_loss += position_loss.item()
                val_angle_loss += angle_loss.item()
                
                # 현재 배치의 검증 손실을 tqdm 진행 막대에 표시
                val_progress.set_postfix({
                    'pos_loss': f'{position_loss.item():.4f}', 
                    'angle_loss': f'{angle_loss.item():.4f}'
                })
        
        # 평균 손실 계산
        avg_train_pos_loss = train_position_loss / len(train_loader)
        avg_train_angle_loss = train_angle_loss / len(train_loader)
        avg_val_pos_loss = val_position_loss / len(val_loader)
        avg_val_angle_loss = val_angle_loss / len(val_loader)
        
        # 학습률 조정
        scheduler.step(avg_val_pos_loss)
        
        # 모델 저장
        current_val_loss = avg_val_pos_loss + 0.5 * avg_val_angle_loss
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, save_path)
            save_msg = f"모델 저장됨: {save_path} (손실: {best_val_loss:.6f})"
        else:
            save_msg = "모델 저장 안됨 (최고 성능 갱신 실패)"
        
        # 에포크 진행 상황 업데이트
        epoch_progress.set_postfix({
            'Train Pos': f'{avg_train_pos_loss:.6f}',
            'Train Angle': f'{avg_train_angle_loss:.6f}',
            'Val Pos': f'{avg_val_pos_loss:.6f}',
            'Val Angle': f'{avg_val_angle_loss:.6f}',
            'Save': save_msg
        })
    
    return model

if __name__ == "__main__":
    # 데이터셋 및 모델 경로 설정
    base_dir = "data"
    model_save_path = "best_trajectory_transformer.pth"
    
    # CUDA 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 데이터셋 로드
    print("데이터셋 준비 중...")
    dataset = TrajectoryDataset(base_dir=base_dir, max_seq_length=100, normalize=True)
    print(f"총 {len(dataset)}개의 궤적 데이터 로드됨")
    
    # 데이터셋 분할 (학습 80%, 검증 20%)
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 데이터 로더 설정
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"데이터셋 분할: 학습 {train_size}개, 검증 {val_size}개")
    
    # 입력 차원 계산
    sample = dataset[0]
    input_dim = sample['features'].shape[1]
    
    # 모델 초기화
    print("모델 초기화 중...")
    model = TrajectoryTransformer(
        input_dim=input_dim,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_length=dataset.max_seq_length,
        num_trajectory_types=len(dataset.type_to_idx)
    )
    model.to(device)
    
    # 모델 학습
    print("모델 학습 시작...")
    epochs = 100
    learning_rate = 0.0001
    
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=learning_rate,
        save_path=model_save_path
    )
    
    print(f"모델 학습 완료! 저장 경로: {model_save_path}")
    
