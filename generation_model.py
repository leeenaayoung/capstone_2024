import numpy as np
import pandas as pd
import torch
import os
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from analyzer import TrajectoryAnalyzer
from scipy.signal import find_peaks, argrelextrema
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class FeatureDetectionModule(nn.Module):
    """궤적의 특징점(피크, 밸리, 변곡점 등)을 감지하는 모듈"""
    def __init__(self, input_dim=4, hidden_dim=16):
        super().__init__()
        
        # 특징점 감지를 위한 1D 컨볼루션 레이어
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1)
        
        # 특징점 중요도 가중치 예측
        self.importance_layer = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        
        # 활성화 함수
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.shape
        
        # 채널 우선으로 변환
        x_trans = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        
        # 특징점 감지
        features = self.relu(self.conv1(x_trans))
        features = self.relu(self.conv2(features))
        feature_map = self.conv3(features)
        
        # 중요도 가중치 (0~1 사이 값)
        importance = self.sigmoid(self.importance_layer(feature_map))
        
        # 원래 형태로 변환
        importance = importance.transpose(1, 2)  # [batch_size, seq_len, input_dim]
        
        return importance

class AdaptiveInterpolationModule(nn.Module):
    """특징점 보존 적응형 보간 모듈"""
    def __init__(self, n_joints=4):
        super().__init__()
        
        # 보간 가중치 예측 네트워크
        self.weight_predictor = nn.Sequential(
            nn.Linear(n_joints * 4, 32),  # 각 관절의 위치와 속도 정보
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, n_joints),
            nn.Sigmoid()  # 0~1 사이의 보간 가중치
        )
    
    def forward(self, source, target, features_source, features_target, t=0.5):
        """
        특징점을 보존하는 적응형 보간 수행
        
        Parameters:
        - source: 소스 궤적 [batch_size, seq_len, n_joints]
        - target: 타겟 궤적 [batch_size, seq_len, n_joints]
        - features_source: 소스 특징점 중요도 [batch_size, seq_len, n_joints]
        - features_target: 타겟 특징점 중요도 [batch_size, seq_len, n_joints]
        - t: 보간 파라미터 (0~1) 사이, 기본값 0.5
        
        Returns:
        - 보간된 궤적 [batch_size, seq_len, n_joints]
        """
        batch_size, seq_len, n_joints = source.shape
        
        # 입력 벡터 생성: [소스 위치, 타겟 위치, 소스 특징점, 타겟 특징점]
        input_vector = torch.cat([
            source, target, features_source, features_target
        ], dim=-1)  # [batch_size, seq_len, n_joints*4]
        
        # 위치별 적응형 보간 가중치 계산
        adaptive_weights = self.weight_predictor(input_vector)  # [batch_size, seq_len, n_joints]
        
        # 전역 보간 파라미터 t와 결합
        final_weights = adaptive_weights * t + (1 - adaptive_weights) * (1 - t)
        
        # 최종 보간 수행
        interpolated = source * (1 - final_weights) + target * final_weights
        
        return interpolated, final_weights

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
    """특징점 인식 및 가중 평균 보간이 통합된 트랜스포머 모델"""
    def __init__(self, input_dim=8, d_model=32, n_head=2, n_layers=4, dropout=0.2):
        super().__init__()
        
        # 기본 입력 처리
        self.input_dim = input_dim  # 입력 차원: 각도 4 + 각속도 4 
        self.d_model = d_model
        self.joint_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # 입력 정규화 및 드롭아웃
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        
        # 특징점 감지 모듈
        self.feature_detector = FeatureDetectionModule(input_dim=4)  # 각도 4개에 대한 특징점
        
        # 적응형 보간 모듈
        self.adaptive_interpolation = AdaptiveInterpolationModule(n_joints=4)
        
        # Joint Attention 레이어
        self.joint_attention_layers = nn.ModuleList([
            JointAttention(d_model) for _ in range(n_layers)
        ])
        
        # TransformerEncoder 레이어
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 2,
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
            nn.Linear(d_model // 2, 4)
        )
    
    def identify_features(self, angles):
        """각도 데이터에서 특징점(피크, 밸리, 변곡점) 식별"""
        batch_size, seq_len, n_joints = angles.shape
        
        # 딥러닝 기반 특징점 감지
        features = self.feature_detector(angles)  # [batch_size, seq_len, n_joints]
        
        return features
    
    def interpolate_trajectories(self, source, target, t=0.5):
        """특징점을 보존하는 궤적 보간 수행"""
        # 각도 데이터 분리
        source_angles = source[:, :, :4]  # [batch_size, seq_len, 4]
        target_angles = target[:, :, :4]  # [batch_size, seq_len, 4]
        
        # 특징점 식별
        source_features = self.identify_features(source_angles)
        target_features = self.identify_features(target_angles)
        
        # 적응형 보간 수행
        interpolated_angles, weights = self.adaptive_interpolation(
            source_angles, target_angles, source_features, target_features, t
        )
        
        # 각속도는 수치 미분으로 계산
        # 간단한 유한 차분법 사용 (향후 개선 가능)
        interpolated_velocities = torch.zeros_like(interpolated_angles)
        for b in range(interpolated_angles.shape[0]):
            for j in range(interpolated_angles.shape[2]):
                interpolated_velocities[b, 1:-1, j] = (
                    interpolated_angles[b, 2:, j] - interpolated_angles[b, :-2, j]
                ) / 2.0
        
        # 첫 번째와 마지막 프레임의 속도는 이웃값 사용
        interpolated_velocities[:, 0, :] = interpolated_velocities[:, 1, :]
        interpolated_velocities[:, -1, :] = interpolated_velocities[:, -2, :]
        
        # 보간된 각도와 각속도 결합
        interpolated_trajectory = torch.cat([
            interpolated_angles, interpolated_velocities
        ], dim=-1)  # [batch_size, seq_len, 8]
        
        return interpolated_trajectory, weights
    
    def forward(self, x, interpolation_mode=False, source=None, target=None, t=0.5):
        """
        전방 전파 함수
        
        Parameters:
        - x: 입력 (일반 모드) 또는 더미 입력 (보간 모드)
        - interpolation_mode: 보간 모드 여부
        - source: 보간 소스 궤적 (보간 모드에서만 사용)
        - target: 보간 타겟 궤적 (보간 모드에서만 사용)
        - t: 보간 파라미터 (보간 모드에서만 사용)
        
        Returns:
        - 예측 또는 보간된 각도 값
        """
        if interpolation_mode and source is not None and target is not None:
            # 보간 모드: 특징점 보존 보간 수행
            interpolated, weights = self.interpolate_trajectories(source, target, t)
            x = interpolated
        
        # x: (batch_size, sequence_length, input_dim=8)
        x = self.joint_embedding(x)  # (batch_size, sequence_length, d_model)
        x = self.positional_encoding(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        
        # 관절 간 상관관계 처리
        joint_features = x
        for attention_layer in self.joint_attention_layers:
            joint_attention = attention_layer(joint_features)
            joint_features = joint_features + joint_attention  # 잔차 연결
        
        # 시퀀스 처리
        x = joint_features.transpose(0, 1)  # (sequence_length, batch_size, d_model)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch_size, sequence_length, d_model)
        
        # 최종 각도 예측
        output = self.output_layer(x)  # (batch_size, sequence_length, 4)
        
        # 보간 모드에서는 가중치도 반환
        if interpolation_mode and source is not None and target is not None:
            return output, weights
        
        return output
    
    def generate_interpolated_trajectory(self, source_trajectory, target_trajectory, t=0.5):
        """
        두 궤적 사이의 특징점 보존 보간 궤적 생성 (추론 전용)
        
        Parameters:
        - source_trajectory: 소스 궤적 [seq_len, 8] (각도 4 + 각속도 4)
        - target_trajectory: 타겟 궤적 [seq_len, 8] (각도 4 + 각속도 4)
        - t: 보간 파라미터 (0~1)
        
        Returns:
        - 보간된 궤적 [seq_len, 8]
        """
        # 배치 차원 추가
        source = torch.FloatTensor(source_trajectory).unsqueeze(0)
        target = torch.FloatTensor(target_trajectory).unsqueeze(0)
        
        # 평가 모드로 전환
        self.eval()
        
        with torch.no_grad():
            # 특징점 보존 보간 수행
            interpolated_angles, weights = self.forward(
                None, interpolation_mode=True, 
                source=source, target=target, t=t
            )
        
        # 배치 차원 제거
        interpolated_angles = interpolated_angles.squeeze(0).cpu().numpy()
        interpolated_trajectory = np.zeros((len(interpolated_angles), 8))
        interpolated_trajectory[:, :4] = interpolated_angles
        
        # 각속도 재계산 (수치 미분)
        for j in range(4):
            interpolated_trajectory[1:-1, 4+j] = (
                interpolated_angles[2:, j] - interpolated_angles[:-2, j]
            ) / 2.0
            interpolated_trajectory[0, 4+j] = interpolated_trajectory[1, 4+j]
            interpolated_trajectory[-1, 4+j] = interpolated_trajectory[-2, 4+j]
        
        return interpolated_trajectory

class FeaturePreservingLoss(nn.Module):
    """특징점 보존 손실 함수"""
    def __init__(self, feature_weight=5.0):
        super().__init__()
        self.base_loss = nn.L1Loss(reduction='none')
        self.feature_weight = feature_weight
    
    def forward(self, pred, target, feature_importance=None):
        """
        특징점에 가중치를 부여한 손실 계산
        
        Parameters:
        - pred: 예측 값 [batch_size, seq_len, n_joints]
        - target: 목표 값 [batch_size, seq_len, n_joints]
        - feature_importance: 특징점 중요도 [batch_size, seq_len, n_joints]
        
        Returns:
        - 가중 손실 값
        """
        # 기본 L1 손실
        base_loss = self.base_loss(pred, target)  # [batch_size, seq_len, n_joints]
        
        if feature_importance is not None:
            # 특징점에 가중치 부여
            weighted_loss = base_loss * (1.0 + self.feature_weight * feature_importance)
        else:
            weighted_loss = base_loss
        
        return weighted_loss.mean()

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

class FeatureAwareTrajectoryDataset(Dataset):
    """특징점 보존 보간이 통합된 데이터셋 클래스"""
    def __init__(self, trajectories, augment=True, num_interpolations=2):
        self.orig_trajectories = []
        self.interp_trajectories = []
        self.interp_weights = []
        
        for traj in trajectories:
            if isinstance(traj, pd.DataFrame):
                angles = traj[['deg1', 'deg2', 'deg3', 'deg4']].values
                velocities = traj[['degsec1', 'degsec2', 'degsec3', 'degsec4']].values
                combined = np.column_stack([angles, velocities])  # (sequence_length, 8)
                self.orig_trajectories.append(torch.FloatTensor(combined))
            else:
                self.orig_trajectories.append(torch.FloatTensor(traj))
        
        # 데이터 증강: 쌍을 만들어 중간 궤적 생성
        if augment and len(self.orig_trajectories) >= 2:
            for _ in range(num_interpolations):
                # 무작위로 두 궤적 선택
                idx1, idx2 = random.sample(range(len(self.orig_trajectories)), 2)
                traj1 = self.orig_trajectories[idx1]
                traj2 = self.orig_trajectories[idx2]
                
                # 두 궤적의 시퀀스 길이가 다르면 더 짧은 쪽에 맞춤
                if traj1.shape[0] != traj2.shape[0]:
                    min_len = min(traj1.shape[0], traj2.shape[0])
                    traj1 = traj1[:min_len]
                    traj2 = traj2[:min_len]
                
                # 무작위 보간 가중치
                t = random.uniform(0.25, 0.75)
                
                # 단순 선형 보간 수행 (후에 모델로 대체됨)
                interp_traj = traj1 * (1 - t) + traj2 * t
                
                self.interp_trajectories.append((traj1, traj2, interp_traj))
                self.interp_weights.append(t)
    
    def __len__(self):
        return len(self.orig_trajectories) + len(self.interp_trajectories)
    
    def __getitem__(self, idx):
        if idx < len(self.orig_trajectories):
            # 원본 궤적 반환
            return self.orig_trajectories[idx], False, None, None, None
        else:
            # 보간 쌍 반환
            interp_idx = idx - len(self.orig_trajectories)
            traj1, traj2, interp_traj = self.interp_trajectories[interp_idx]
            t = self.interp_weights[interp_idx]
            return interp_traj, True, traj1, traj2, t

class GenerationModel:
    """특징점 인식 및 가중 평균 보간이 통합된 모델 훈련 및 생성 클래스"""
    def __init__(self, base_dir=None, model_save_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_dir = base_dir or os.path.join(os.getcwd(), "data")

        if model_save_path:
            self.model_path = model_save_path
        else:
            self.model_path = os.path.join(os.getcwd(), "feature_aware_model.pth")
        
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
                        
                        # 각속도 계산 (이미 있으면 사용, 없으면 계산)
                        if all(col in df.columns for col in ['degsec1', 'degsec2', 'degsec3', 'degsec4']):
                            velocities = df[['degsec1', 'degsec2', 'degsec3', 'degsec4']].values
                        else:
                            velocities = np.zeros_like(angles)
                            for j in range(4):
                                velocities[1:-1, j] = (angles[2:, j] - angles[:-2, j]) / 2.0
                                velocities[0, j] = velocities[1, j]
                                velocities[-1, j] = velocities[-2, j]
                        
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
    
    def generate_feature_enhanced_data(self, trajectories, num_interpolations=5):
        """특징점 기반 보간을 통한 데이터 증강"""
        enhanced_trajectories = trajectories.copy()
        
        if len(trajectories) < 2:
            return enhanced_trajectories
        
        n_trajectories = len(trajectories)
        for _ in range(num_interpolations):
            # 무작위로 두 궤적 선택
            idx1, idx2 = random.sample(range(n_trajectories), 2)
            traj1 = trajectories[idx1]
            traj2 = trajectories[idx2]
            
            # 두 궤적의 시퀀스 길이가 다르면 더 짧은 쪽에 맞춤
            if len(traj1) != len(traj2):
                min_len = min(len(traj1), len(traj2))
                traj1 = traj1[:min_len]
                traj2 = traj2[:min_len]
            
            # 특징점 검출 (NumPy 버전)
            peaks = []
            valleys = []
            inflections = []
            
            for j in range(4):  # 각 관절에 대해
                # 피크 및 밸리 감지
                peak_indices, _ = find_peaks(traj1[:, j])
                valley_indices, _ = find_peaks(-traj1[:, j])
                
                # 변곡점 감지 (2차 도함수의 부호 변화)
                d2 = np.gradient(np.gradient(traj1[:, j]))
                infl_indices = []
                for i in range(1, len(d2)):
                    if d2[i-1] * d2[i] < 0:
                        infl_indices.append(i)
                
                peaks.append(peak_indices)
                valleys.append(valley_indices)
                inflections.append(infl_indices)
            
            # 특징점 가중치 생성
            feature_weights = np.ones((len(traj1), 4)) * 0.5  # 기본 가중치 0.5
            
            for j in range(4):
                # 피크와 밸리 주변 가중치 증가
                for idx in np.concatenate([peaks[j], valleys[j]]):
                    start = max(0, idx - 3)
                    end = min(len(traj1), idx + 4)
                    # 가우시안형 가중치 적용 (피크/밸리에 가까울수록 1에 가깝게)
                    for i in range(start, end):
                        dist = abs(i - idx)
                        feature_weights[i, j] = min(1.0, feature_weights[i, j] + np.exp(-0.5 * dist**2))
            
            # 샘플 가중치 보간
            sample_ts = [0.2, 0.35, 0.5, 0.65, 0.8]
            for t in sample_ts:
                # 특징점 보존 보간 (간단한 구현)
                interp_traj = np.zeros_like(traj1)
                
                for j in range(4):  # 각 관절에 대해
                    for i in range(len(traj1)):
                        # 적응형 가중치 (특징점에 따라 조정)
                        adaptive_weight = feature_weights[i, j] * t + (1 - feature_weights[i, j]) * (1 - t)
                        interp_traj[i, j] = traj1[i, j] * (1 - adaptive_weight) + traj2[i, j] * adaptive_weight
                
                # 각속도 재계산
                for j in range(4):
                    interp_traj[1:-1, 4+j] = (interp_traj[2:, j] - interp_traj[:-2, j]) / 2.0
                    interp_traj[0, 4+j] = interp_traj[1, 4+j]
                    interp_traj[-1, 4+j] = interp_traj[-2, 4+j]
                
                enhanced_trajectories.append(interp_traj)
        
        return enhanced_trajectories
    
    def train_model(self, trajectories=None, epochs=100, batch_size=32, learning_rate=0.001, feature_weight=2.0):
        """특징점 보존 보간이 통합된 모델 학습"""
        if trajectories is None:
            trajectories = self.collect_training_data()
            
        if not trajectories:
            print("No training data available. Skipping model training.")
            return False
        
        # 특징점 기반 보간으로 데이터 증강
        print("Enhancing training data with feature-based interpolation...")
        enhanced_trajectories = self.generate_feature_enhanced_data(trajectories)
        print(f"Original trajectories: {len(trajectories)}, Enhanced: {len(enhanced_trajectories)}")
        
        # 데이터셋 준비 (원본 + 보간 데이터)
        dataset = FeatureAwareTrajectoryDataset(enhanced_trajectories)
        
        def collate_fn(batch):
            # 배치의 각 항목 분리
            trajs = []
            is_interp_flags = []
            source_trajs = []
            target_trajs = []
            interp_params = []
            
            # 최대 시퀀스 길이 계산
            max_len = max(x[0].shape[0] for x in batch)
            
            for traj, is_interp, source, target, t in batch:
                # 패딩 적용
                if len(traj) < max_len:
                    pad = torch.zeros((max_len - len(traj), traj.shape[1]))
                    traj_padded = torch.cat([traj, pad], dim=0)
                else:
                    traj_padded = traj[:max_len]
                
                trajs.append(traj_padded)
                is_interp_flags.append(is_interp)
                
                if is_interp:
                    # 보간 데이터의 경우 소스/타겟 궤적도 패딩
                    if len(source) < max_len:
                        pad = torch.zeros((max_len - len(source), source.shape[1]))
                        source_padded = torch.cat([source, pad], dim=0)
                        target_padded = torch.cat([target, pad], dim=0)
                    else:
                        source_padded = source[:max_len]
                        target_padded = target[:max_len]
                    
                    source_trajs.append(source_padded)
                    target_trajs.append(target_padded)
                    interp_params.append(t)
                else:
                    # 원본 데이터는 더미 값 추가
                    source_trajs.append(None)
                    target_trajs.append(None)
                    interp_params.append(None)
            
            # 텐서로 변환
            trajs = torch.stack(trajs)
            is_interp_flags = torch.tensor(is_interp_flags)
            
            return trajs, is_interp_flags, source_trajs, target_trajs, interp_params
        
        train_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        # 모델 훈련 모드로 설정
        self.model.train()
        
        # 손실 함수 및 옵티마이저 설정
        criterion = FeaturePreservingLoss(feature_weight=feature_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        best_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        
        print(f"Starting feature-aware joint trajectory model training...")
        
        for epoch in range(epochs):
            epoch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            total_loss = 0.0
            
            for batch_data in epoch_progress:
                trajs, is_interp_flags, source_trajs, target_trajs, interp_params = batch_data
                
                # GPU로 전송 (사용 가능한 경우)
                trajs = trajs.to(self.device)
                is_interp_flags = is_interp_flags.to(self.device)
                
                # 배치를 원본 데이터와 보간 데이터로 분리
                orig_mask = ~is_interp_flags
                interp_mask = is_interp_flags
                
                # 옵티마이저 그래디언트 초기화
                optimizer.zero_grad()
                
                # Forward 패스 (원본 데이터)
                if orig_mask.sum() > 0:
                    orig_trajs = trajs[orig_mask]
                    # 입력: 각도 + 각속도 (8차원), 타겟: 각도만 (4차원)
                    output_orig = self.model(orig_trajs)
                    target_orig = orig_trajs[:, :, :4]
                    
                    # 특징점 감지
                    feature_importance = self.model.identify_features(target_orig)
                    
                    # 특징점 보존 손실 계산
                    loss_orig = criterion(output_orig, target_orig, feature_importance)
                else:
                    loss_orig = 0.0
                
                # Forward 패스 (보간 데이터)
                if interp_mask.sum() > 0:
                    interp_trajs = trajs[interp_mask]
                    
                    # 보간 데이터의 소스와 타겟 준비
                    interp_indices = torch.where(interp_mask)[0]
                    batch_sources = []
                    batch_targets = []
                    batch_ts = []
                    
                    for idx in interp_indices:
                        src = source_trajs[idx]
                        tgt = target_trajs[idx]
                        t = interp_params[idx]
                        
                        if src is not None and tgt is not None:
                            batch_sources.append(src)
                            batch_targets.append(tgt)
                            batch_ts.append(t)
                    
                    if batch_sources:  # 보간 데이터가 있는 경우에만 처리
                        batch_sources = torch.stack(batch_sources).to(self.device)
                        batch_targets = torch.stack(batch_targets).to(self.device)
                        batch_ts = torch.tensor(batch_ts, device=self.device)
                        
                        # 보간 모드로 예측
                        output_interp, weights = self.model(
                            None, 
                            interpolation_mode=True,
                            source=batch_sources,
                            target=batch_targets,
                            t=batch_ts.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1] 형태로 변환
                        )
                        
                        # 타겟은 보간 궤적의 각도 부분
                        target_interp = interp_trajs[:, :, :4]
                        
                        # 보간 손실 계산
                        loss_interp = criterion(output_interp, target_interp)
                    else:
                        loss_interp = 0.0
                else:
                    loss_interp = 0.0
                
                # 총 손실 계산
                if orig_mask.sum() > 0 and interp_mask.sum() > 0:
                    loss = loss_orig * 0.7 + loss_interp * 0.3  # 원본:보간 = 7:3 비율
                elif orig_mask.sum() > 0:
                    loss = loss_orig
                elif interp_mask.sum() > 0:
                    loss = loss_interp
                else:
                    continue  # 유효한 데이터가 없는 경우 스킵
                
                # 역전파 및 옵티마이저 스텝
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                epoch_progress.set_postfix(loss=f"{loss.item():.4f}")
            
            # 에폭 평균 손실 계산
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')
            
            # 학습률 스케줄러 업데이트
            scheduler.step(avg_loss)
            
            # 최적 모델 저장
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = self.model.state_dict().copy()
                best_epoch = epoch + 1
                print(f"New best model found at epoch {best_epoch} with loss: {best_loss:.4f}")
        
        # 최적 모델 복원 및 저장
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
        
        # 평가 모드로 전환
        self.model.eval()
        return True