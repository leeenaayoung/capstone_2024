import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import random
from utils import *
from generation import TrajectoryGenerator
from evaluate import TrajectoryAnalyzer

def load_and_preprocess_data(base_dir):
    """all_data 디렉토리의 모든 데이터를 로드하고 타입별로 분류"""
    trajectories_by_type = {
        'line': [],
        'arc': [],
        'circle': []
    }
    
    print("\nLoading and classifying trajectories...")
    for file in os.listdir(base_dir):
        if file.endswith('.txt'):
            file_path = os.path.join(base_dir, file)
            
            # 궤적 타입 결정
            if 'd_' in file:
                traj_type = 'line'
            elif 'v_' in file or 'h_' in file:
                traj_type = 'arc'
            elif 'clock_' in file or 'counter_' in file:
                traj_type = 'circle'
            else:
                continue
            
            # 데이터 로드 및 전처리
            raw_data = pd.read_csv(file_path)
            processed_data = preprocess_trajectory_data(raw_data)
            
            # deg 데이터만 추출
            deg_data = processed_data[['deg1', 'deg2', 'deg3', 'deg4']].values
            print(deg_data[:5])
            trajectories_by_type[traj_type].append(deg_data)
            
    
    # 타입별 데이터 개수 출력
    for traj_type, trajs in trajectories_by_type.items():
        print(f"Loaded {len(trajs)} {traj_type} trajectories")
    
    return trajectories_by_type

def analyze_joint_correlations(trajectories_by_type):
    correlations = {}
    
    for traj_type, trajs in trajectories_by_type.items():
        print(f"\nAnalyzing correlations for {traj_type} trajectories...")
        
        # 데이터 존재 여부 및 상세 정보 확인
        if not trajs:
            print(f"No trajectories found for {traj_type}")
            continue
        
        print(f"Number of trajectories: {len(trajs)}")
        
        # 각 궤적의 형태와 크기 출력
        for i, traj in enumerate(trajs):
            print(f"Trajectory {i} shape: {traj.shape}")
            print(f"Sample of trajectory {i}:")
            print(traj[:5])  # 각 궤적의 첫 5개 데이터 포인트 출력
        
        # 모든 궤적의 각도 데이터 결합
        try:
            all_angles = np.concatenate(trajs, axis=0)
            print("\nConcatenated data shape:", all_angles.shape)
            
            # 각도 데이터의 기본 통계 출력
            print("\nBasic statistics:")
            print("Mean:", np.mean(all_angles, axis=0))
            print("Std:", np.std(all_angles, axis=0))
            print("Min:", np.min(all_angles, axis=0))
            print("Max:", np.max(all_angles, axis=0))
            
            # 상관관계 계산
            corr_matrix = np.corrcoef(all_angles.T)
            correlations[traj_type] = corr_matrix
            
            # 상관관계 행렬 출력
            print("\nCorrelation Matrix:")
            print(corr_matrix)
            
            # 나머지 기존 시각화 코드 유지
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                pd.DataFrame(
                    corr_matrix,
                    columns=['deg1', 'deg2', 'deg3', 'deg4'],
                    index=['deg1', 'deg2', 'deg3', 'deg4']
                ),
                annot=True,
                cmap='coolwarm',
                vmin=-1,
                vmax=1
            )
            plt.title(f'Joint Angle Correlations - {traj_type.upper()} Trajectories')
            plt.tight_layout()
            
            # 이미지 저장
            os.makedirs('correlation_matrices', exist_ok=True)
            plt.savefig(f'correlation_matrices/{traj_type}_correlations.png')
            plt.close()  # plt.show() 대신 plt.close() 사용
            
        except Exception as e:
            print(f"Error processing trajectories: {e}")
    
    return correlations

class JointAttention(nn.Module):
    """Joint 간의 관계를 학습하는 Self-Attention 모듈"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Scaled dot-product attention
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

class JointTrajectoryDataset(Dataset):
    def __init__(self, trajectories_by_type, seq_length=1000):  # 기본 길이 200으로 설정
        self.data = []
        
        for trajs in trajectories_by_type.values():
            for traj in trajs:
                # 리샘플링으로 모든 궤적을 동일한 길이로 만듦
                indices = np.linspace(0, len(traj)-1, seq_length, dtype=int)
                resampled_traj = traj[indices]
                self.data.append(torch.FloatTensor(resampled_traj))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def create_data_loader(trajectories_by_type, batch_size=32):
    dataset = JointTrajectoryDataset(trajectories_by_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class JointTrajectoryTransformer(nn.Module):
    def __init__(self, n_joints=4, d_model=64, n_head=4, n_layers=3, dropout=0.1):
        super().__init__()
        
        self.joint_embedding = nn.Linear(n_joints, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # 입력 처리를 위한 레이어
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        
        # Joint 간 관계를 학습하는 attention layers
        self.joint_attention_layers = nn.ModuleList([
            JointAttention(d_model) for _ in range(n_layers)
        ])
        
        # Transformer encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model*4,
            dropout=dropout
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
        # x shape: (batch_size, seq_len, n_joints)
        x = self.joint_embedding(x)
        x = self.positional_encoding(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        
        # Joint attention
        joint_features = x
        for attention_layer in self.joint_attention_layers:
            joint_attention = attention_layer(joint_features)
            joint_features = joint_features + joint_attention
        
        # Transformer
        x = joint_features.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        
        # 출력 생성
        output = self.output_layer(x)
        
        return output

def train_model(model, train_loader, device, epochs=100):
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            output = model(batch)
            
            # Loss 계산
            reconstruction_loss = criterion(output, batch)
            smoothness_loss = torch.mean(torch.diff(output, dim=1)**2)
            
            # Joint 간 상관관계 loss
            reshaped_output = output.transpose(1, 2).reshape(output.size(-1), -1)
            joint_correlation = torch.corrcoef(reshaped_output)
            correlation_loss = torch.mean((joint_correlation - torch.eye(4, device=device))**2)
            
            loss = reconstruction_loss + 0.1 * smoothness_loss + 0.01 * correlation_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

def main():
    base_dir = os.path.join(os.getcwd(), "data/all_data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # 데이터 로드 및 전처리
        print("\nLoading and preprocessing data...")
        trajectories_by_type = load_and_preprocess_data(base_dir)
        
        # 상관관계 분석
        print("\nAnalyzing joint correlations...")
        correlations = analyze_joint_correlations(trajectories_by_type)
        
        # 데이터로더 생성
        print("\nPreparing data loader...")
        train_loader = create_data_loader(trajectories_by_type)
        
        # 모델 초기화 및 학습
        print("\nInitializing and training model...")
        model = JointTrajectoryTransformer().to(device)
        train_model(model, train_loader, device)
        
        # 모델 저장
        print("\nSaving model...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'correlations': correlations
        }, 'generate_model.pth')
        
        # 테스트
        print("\nTesting model with interpolated trajectory...")
        
        # TrajectoryGenerator 초기화
        analyzer = TrajectoryAnalyzer(
            classification_model="best_classification_model.pth",
            base_dir="data"
        )
        generator = TrajectoryGenerator(analyzer)
        
        # 테스트용 궤적 선택
        test_dir = os.path.join("data/non_golden_sample")
        test_files = [f for f in os.listdir(test_dir) if f.endswith('.txt')]
        
        if not test_files:
            raise ValueError("No test files found.")
        
        # 테스트 파일 선택 및 처리
        selected_file = random.choice(test_files)
        file_path = os.path.join(test_dir, selected_file)
        
        print(f"\nTesting with file: {selected_file}")
        
        # 궤적 생성
        user_trajectory, trajectory_type = analyzer.load_user_trajectory(file_path)
        target_trajectory, _ = analyzer.load_target_trajectory(trajectory_type)
        
        # Generator의 interpolate_trajectory 사용
        generated_df = generator.interpolate_trajectory(
            target_df=target_trajectory,
            user_df=user_trajectory,
            trajectory_type=trajectory_type
        )
        
        # Transformer 모델로 처리
        model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(
                generated_df[['deg1', 'deg2', 'deg3', 'deg4']].values
            ).unsqueeze(0).to(device)
            
            output_tensor = model(input_tensor)
            transformed_angles = output_tensor.squeeze(0).cpu().numpy()
        
        # 결과 생성
        final_generated_df = generated_df.copy()
        final_generated_df[['deg1', 'deg2', 'deg3', 'deg4']] = transformed_angles
        
        # End-effector 위치 업데이트
        endeffector_degrees = transformed_angles.copy()
        endeffector_degrees[:, 1] -= 90
        endeffector_degrees[:, 3] -= 90
        
        aligned_points = np.array([calculate_end_effector_position(deg) for deg in endeffector_degrees])
        aligned_points = aligned_points * 1000
        
        final_generated_df[['x_end', 'y_end', 'z_end']] = aligned_points
        
        # 결과 시각화
        print("\nVisualizing results...")
        generator.visualize_trajectories(
            target_df=target_trajectory,
            user_df=user_trajectory,
            generated_df=final_generated_df,
            trajectory_type=trajectory_type
        )
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()