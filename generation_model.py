import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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

class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, trajectories):
        """
        궤적 데이터셋 초기화
        
        Args:
            trajectories (list): 궤적 데이터 리스트
        """
        # 모든 궤적을 텐서로 변환
        self.data = [torch.FloatTensor(traj) for traj in trajectories]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def generate_sample_trajectories(num_samples=1000, seq_length=500):
    """
    간단한 샘플 궤적 생성
    
    Args:
        num_samples (int): 생성할 궤적 수
        seq_length (int): 각 궤적의 길이
    
    Returns:
        list: 생성된 궤적 리스트
    """
    trajectories = []
    
    for _ in range(num_samples):
        # 각 관절에 대해 다른 패턴의 궤적 생성
        traj = np.zeros((seq_length, 4))
        
        # 사인파, 코사인파, 선형, 지수 함수 등 다양한 패턴 생성
        traj[:, 0] = np.sin(np.linspace(0, 10*np.pi, seq_length)) * 45  # 사인파
        traj[:, 1] = np.cos(np.linspace(0, 10*np.pi, seq_length)) * 30  # 코사인파
        traj[:, 2] = np.linspace(0, 60, seq_length)  # 선형 증가
        traj[:, 3] = 15 + np.exp(np.linspace(0, 2, seq_length)) * 10  # 지수 함수
        
        trajectories.append(traj)
    
    return trajectories

def train_trajectory_model(
    model, 
    train_loader, 
    epochs=100, 
    learning_rate=0.001, 
    device=None
):
    """
    궤적 생성 모델 학습
    
    Args:
        model (nn.Module): 학습할 모델
        train_loader (DataLoader): 학습 데이터 로더
        epochs (int): 학습 에포크 수
        learning_rate (float): 학습률
        device (torch.device, optional): 학습에 사용할 장치
    """
    from tqdm import tqdm
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.train()
    
    # 손실 함수와 옵티마이저 설정
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 학습 루프 (전체 에포크에 대한 프로그레스 바)
    for epoch in tqdm(range(epochs), desc="Training Progress", position=0):
        total_loss = 0.0
        
        # 배치 단위 프로그레스 바
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}", position=1, leave=False)
        
        for batch in batch_iterator:
            batch = batch.to(device)
            
            # 순전파
            optimizer.zero_grad()
            output = model(batch)
            
            # 손실 계산
            loss = criterion(output, batch)
            
            # 역전파 및 최적화
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 배치별 손실 표시
            batch_iterator.set_postfix(loss=loss.item())
        
        # 에포크별 평균 손실 출력
        tqdm.write(f'Epoch [{epoch+1}/{epochs}], Avg Loss: {total_loss/len(train_loader):.4f}')

def visualize_trajectories(original_trajs, generated_trajs):
    """
    원본 및 생성된 궤적 시각화
    
    Args:
        original_trajs (list): 원본 궤적 리스트
        generated_trajs (list): 생성된 궤적 리스트
    """
    # 랜덤하게 몇 개의 궤적 선택하여 시각화
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Original vs Generated Trajectories')
    
    for i in range(2):
        trajs = original_trajs if i == 0 else generated_trajs
        title = 'Original' if i == 0 else 'Generated'
        
        for j in range(4):
            # 첫 번째 궤적 선택
            traj = trajs[0]
            axs[i, j].plot(traj[:, j])
            axs[i, j].set_title(f'{title} Joint {j+1}')
            axs[i, j].set_xlabel('Time Step')
            axs[i, j].set_ylabel('Angle')
    
    plt.tight_layout()
    plt.show()

def main():
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 샘플 궤적 생성
    print("Generating sample trajectories...")
    trajectories = generate_sample_trajectories()
    
    # 데이터셋 및 데이터로더 생성
    dataset = TrajectoryDataset(trajectories)
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True
    )
    
    # 모델 초기화
    print("Initializing trajectory generation model...")
    model = JointTrajectoryTransformer().to(device)
    
    # 모델 학습
    print("Training model...")
    train_trajectory_model(model, train_loader, device=device)
    
    # 모델 저장
    print("Saving model...")
    torch.save({
        'model_state_dict': model.state_dict()
    }, 'trajectory_generation_model.pth')
    
    # 새로운 궤적 생성 테스트
    print("Generating new trajectories...")
    model.eval()
    generated_trajs = []
    
    with torch.no_grad():
        for original_traj in trajectories[:5]:  # 처음 5개 궤적을 기반으로 생성
            input_tensor = torch.FloatTensor(original_traj).unsqueeze(0).to(device)
            generated_traj = model(input_tensor).squeeze(0).cpu().numpy()
            generated_trajs.append(generated_traj)
    
    # 궤적 시각화
    visualize_trajectories(trajectories[:5], generated_trajs)

if __name__ == "__main__":
    main()