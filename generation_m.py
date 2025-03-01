import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from fastdtw import fastdtw
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import euclidean
import os
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from analyzer import TrajectoryAnalyzer
from utils import calculate_end_effector_position

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
    """관절 간 상관관계를 학습하는 트랜스포머 모델"""
    def __init__(self, n_joints=4, d_model=64, n_head=4, n_layers=3, dropout=0.1):
        super().__init__()
        
        self.joint_embedding = nn.Linear(n_joints, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # 입력 레이어
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.joint_attention_layers = nn.ModuleList([
            JointAttention(d_model) for _ in range(n_layers)
        ])
        
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
        x = self.joint_embedding(x)
        x = self.positional_encoding(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        joint_features = x
        for attention_layer in self.joint_attention_layers:
            joint_attention = attention_layer(joint_features)
            joint_features = joint_features + joint_attention

        x = joint_features.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        
        # 출력 생성
        output = self.output_layer(x)
        
        return output

class TrajectoryDataset(Dataset):
    """궤적 데이터셋 클래스"""
    def __init__(self, trajectories):
        self.data = [torch.FloatTensor(traj) for traj in trajectories]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class ModelBasedTrajectoryGenerator:
    """모델 기반 궤적 생성기 클래스"""
    def __init__(self, analyzer, model_path=None):
        self.analyzer = analyzer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 관절 상관 관계 모델 초기화
        self.model = JointTrajectoryTransformer().to(self.device)
        
        # 관절 제한 설정
        self.joint_limits = {   
            0: (-10, 110),
            1: (0, 150),     
            2: (0, 150),     
            3: (-90, 90)     
        }
        
        # 기본적으로 모델을 평가 모드로 설정
        self.model.eval()
    
    def collect_training_data(self, data_dir=None, n_samples=100):
        """ 학습 데이터 수집 """
        base_dir = data_dir or os.path.join(self.analyzer.base_dir, "golden_sample")
        trajectories = []
        
        try:
            trajectory_files = [f for f in os.listdir(base_dir) if f.endswith('.txt')]
            
            if len(trajectory_files) > n_samples:
                trajectory_files = random.sample(trajectory_files, n_samples)
            
            print(f"총 {len(trajectory_files)}개의 궤적 파일에서 데이터 수집 중...")
            
            for file_name in tqdm(trajectory_files, desc="파일 로드 중"):
                file_path = os.path.join(base_dir, file_name)
                
                # 궤적 유형 추출
                trajectory_type = None
                for type_name in ['line', 'clockwise', 'counter_clockwise', 'v_shape', 'h_shape']:
                    if type_name in file_name.lower():
                        trajectory_type = type_name
                        break
                
                if trajectory_type:
                    try:
                        # TrajectoryAnalyzer에서 궤적 로드
                        df, _ = self.analyzer.load_target_trajectory(trajectory_type)
                        
                        # 각도 데이터 추출
                        angles = df[['deg1', 'deg2', 'deg3', 'deg4']].values
                        
                        # 궤적이 너무 짧지 않은 경우에만 추가
                        if len(angles) >= 30:
                            trajectories.append(angles)
                    except Exception as e:
                        print(f"파일 {file_name} 처리 중 오류 발생: {str(e)}")
            
            print(f"총 {len(trajectories)}개의 궤적 데이터 수집 완료")
            
        except Exception as e:
            print(f"데이터 수집 중 오류 발생: {str(e)}")
        
        return trajectories
    
    def train_model(self, trajectories=None, epochs=100, batch_size=32, learning_rate=0.001):
        """ 관절 관계 모델 학습 """
        # 데이터가 제공되지 않은 경우 자동 수집
        if trajectories is None:
            trajectories = self.collect_training_data()
            
        if not trajectories:
            print("학습 데이터가 없습니다. 모델 학습을 건너뜁니다.")
            return
            
        # 데이터셋 및 데이터로더 생성
        dataset = TrajectoryDataset(trajectories)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 모델을 학습 모드로 설정
        self.model.train()
        
        # 손실 함수 및 옵티마이저 설정
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 학습 루프
        print(f"관절 관계 모델 학습 시작 (총 {len(trajectories)}개 궤적)...")
        for epoch in tqdm(range(epochs), desc="학습 진행 상황"):
            total_loss = 0.0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                # 순전파
                optimizer.zero_grad()
                output = self.model(batch)
                
                # 손실 계산 - 관절 간 관계를 학습하기 위해 원본 궤적을 예측하도록 학습
                loss = criterion(output, batch)
                
                # 역전파 및 최적화
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # 에포크별 평균 손실 출력
            print(f'Epoch [{epoch+1}/{epochs}], 평균 손실: {total_loss/len(train_loader):.4f}')
            
        # 학습 후, 모델 저장
        model_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(model_dir, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, os.path.join(model_dir, 'joint_relationship_model.pth'))
        
        print("모델 학습 완료 및 저장됨.")
        
        # 모델을 다시 평가 모드로 설정
        self.model.eval()
    
    def smooth_data(self, data, R=0.02, Q=0.1):
        """칼만 필터를 사용한 데이터 스무딩"""
        # 각도와 각속도 데이터 추출
        angles = data[['deg1', 'deg2', 'deg3', 'deg4']].values
        velocities = data[['degsec1', 'degsec2', 'degsec3', 'degsec4']].values
        n_samples, n_joints = angles.shape
        
        # 결과를 저장할 배열 초기화
        smoothed_angles = np.zeros_like(angles)
        smoothed_velocities = np.zeros_like(velocities)
        
        for joint in range(n_joints):   
            # 초기 상태 설정
            x_hat_full = np.array([angles[0, joint], velocities[0, joint]])
            P_full = np.eye(2)
            
            # 시스템 행렬 설정
            dt = 1.0
            A = np.array([[1, dt],
                        [0, 1]])
            H = np.eye(2) 
            
            # 프로세스 노이즈 설정
            Q_matrix = Q * np.array([[dt**4/4, dt**3/2],
                                    [dt**3/2, dt**2]])
            # 측정 노이즈 설정
            R_matrix = np.diag([R, R*10]) 
            
            # 첫 번째 상태 저장
            smoothed_angles[0, joint] = x_hat_full[0]
            smoothed_velocities[0, joint] = x_hat_full[1]
            
            # 칼만 필터 적용
            for k in range(1, n_samples):
                # 예측 단계
                x_hat_full = A @ x_hat_full
                P_full = A @ P_full @ A.T + Q_matrix
                
                # 현재 측정값
                z = np.array([angles[k, joint], velocities[k, joint]])
                
                # 업데이트 단계
                y = z - H @ x_hat_full
                S = H @ P_full @ H.T + R_matrix
                K = P_full @ H.T @ np.linalg.inv(S)
                
                # 상태 업데이트
                x_hat_full = x_hat_full + K @ y
                P_full = (np.eye(2) - K @ H) @ P_full
                
                # 결과 저장
                smoothed_angles[k, joint] = x_hat_full[0]
                smoothed_velocities[k, joint] = x_hat_full[1]
        
        # 결과를 데이터프레임으로 변환
        smoothed_df = data.copy()
        smoothed_df[['deg1', 'deg2', 'deg3', 'deg4']] = smoothed_angles
        smoothed_df[['degsec1', 'degsec2', 'degsec3', 'degsec4']] = smoothed_velocities
        
        return smoothed_df

    def normalize_time(self, target_trajectory, subject_trajectory):
        """궤적을 시간에 대해 정규화"""
        target_angles = target_trajectory[:, :4]
        subject_angles = subject_trajectory[:, :4]
        
        # 각도 데이터 정규화
        normalized_target_angles = np.zeros_like(target_angles)
        normalized_subject_angles = np.zeros_like(subject_angles)
        
        for joint in range(4):
            min_val = self.joint_limits[joint][0] 
            max_val = self.joint_limits[joint][1] 
            range_val = max_val - min_val
            normalized_target_angles[:, joint] = (target_angles[:, joint] - min_val) / range_val
            normalized_subject_angles[:, joint] = (subject_angles[:, joint] - min_val) / range_val
        
        # DTW로 정렬
        _, path = fastdtw(normalized_target_angles, normalized_subject_angles, dist=euclidean)
        path = np.array(path, dtype=np.int32)
        
        # 찾은 경로로 전체 궤적 정렬 (각도 + 각속도)
        aligned_target = target_trajectory[path[:, 0]]
        aligned_subject = subject_trajectory[path[:, 1]]
        
        return aligned_target, aligned_subject
        
    def model_based_interpolate_line(self, target, subject, interpolation_weight=0.5):
        """관절 관계를 고려한 모델 기반 선형 보간"""
        # 데이터 분리
        target_angles = target[:, :4]  
        target_velocities = target[:, 4:]
        subject_angles = subject[:, :4]
        subject_velocities = subject[:, 4:]

        # DTW를 통한 시간 정렬
        aligned_target, aligned_subject = self.normalize_time(target, subject)
        
        # 정렬된 데이터 분리
        aligned_target_angles = aligned_target[:, :4]
        aligned_target_velocities = aligned_target[:, 4:]
        aligned_subject_angles = aligned_subject[:, :4]
        aligned_subject_velocities = aligned_subject[:, 4:]

        # 결과 저장을 위한 배열 초기화
        interpolated_degrees = np.zeros_like(aligned_target_angles)
        interpolated_velocities = np.zeros_like(aligned_target_velocities)

        # 기본 선형 보간 수행
        for joint in range(4):
            for i in range(len(aligned_target_angles)):
                # 직접적인 선형 보간(항상 두 값 사이의 결과를 보장)
                interpolated_degrees[i, joint] = (1 - interpolation_weight) * aligned_target_angles[i, joint] + \
                                               interpolation_weight * aligned_subject_angles[i, joint]
                
                interpolated_velocities[i, joint] = (1 - interpolation_weight) * aligned_target_velocities[i, joint] + \
                                                  interpolation_weight * aligned_subject_velocities[i, joint]

        # 모델 기반 관절 간 상호작용 적용
        with torch.no_grad():
            n_points = len(interpolated_degrees)
            
            # 긴 시퀀스는 세그먼트로 나누어 처리
            segments = []
            segment_size = 100
            
            for i in range(0, n_points, segment_size):
                # 세그먼트 추출 및 처리
                segment = interpolated_degrees[i:i+segment_size]
                if len(segment) == 0:
                    continue
                    
                # 모델 입력 준비 및 처리
                segment_tensor = torch.FloatTensor(segment).unsqueeze(0).to(self.device)
                joint_interactions = self.model(segment_tensor).squeeze(0).cpu().numpy()
                
                # 보정된 세그먼트 저장
                segments.append(joint_interactions)
            
            # 세그먼트 결합 및 모델 출력 처리
            if segments:
                model_output = np.vstack(segments)
                if len(model_output) > len(interpolated_degrees):
                    model_output = model_output[:len(interpolated_degrees)]
                
                # 모델 출력과 기본 보간 결과 블렌딩
                correction_strength = 0.3 
                for joint in range(4):
                    for i in range(len(interpolated_degrees)):
                        lower_bound = min(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                        upper_bound = max(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])

                        bounded_model_output = np.clip(model_output[i, joint], lower_bound, upper_bound)

                        original_val = interpolated_degrees[i, joint]
                        interpolated_degrees[i, joint] = (1 - correction_strength) * original_val + correction_strength * bounded_model_output

                        interpolated_degrees[i, joint] = np.clip(interpolated_degrees[i, joint], lower_bound, upper_bound)

        # 수정된 각도를 기반으로 각속도 재계산
        for joint in range(4):
            interpolated_velocities[:, joint] = np.gradient(interpolated_degrees[:, joint])

        return interpolated_degrees, interpolated_velocities
    
    def model_based_interpolate_arc(self, target, subject):
        """관절 관계를 고려한 모델 기반 호 보간"""
        # 시간 정규화
        aligned_target, aligned_subject = self.normalize_time(target, subject)
        
        # 데이터 분리
        aligned_target_angles = aligned_target[:, :4]
        aligned_target_velocities = aligned_target[:, 4:]
        aligned_subject_angles = aligned_subject[:, :4]
        aligned_subject_velocities = aligned_subject[:, 4:]

        n_points = len(aligned_target_angles)
        interpolated_degrees = np.zeros_like(aligned_target_angles)
        interpolated_velocities = np.zeros_like(aligned_target_velocities)
        
        # 시간 포인트
        t = np.linspace(0, 1, n_points)
        interpolation_weight = 0.5  # 기본 보간 가중치
        
        # 1단계: 기본 Cubic Spline 보간 수행
        for joint in range(4):
            # 먼저 기본 선형 보간 계산 (나중에 경계로 사용)
            for i in range(n_points):
                # 각 시점의 선형 보간된 값 (경계 역할)
                lower_bound = min(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                upper_bound = max(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                
                # 3개의 제어점 사용
                control_indices = [0, n_points//2, -1]
                control_times = [0, 0.5, 1]
                
                # 제어점의 각도와 각속도 계산
                control_angles = np.zeros(len(control_indices))
                control_velocities = np.zeros(len(control_indices))
                
                for i, idx in enumerate(control_indices):
                    # 제어점도 항상 두 궤적 사이에 있도록 보장
                    target_val = aligned_target_angles[idx, joint]
                    subject_val = aligned_subject_angles[idx, joint]
                    ctrl_lower = min(target_val, subject_val)
                    ctrl_upper = max(target_val, subject_val)
                    
                    # 제어점 계산 및 경계 내로 제한
                    control_angles[i] = np.clip(
                        (1 - interpolation_weight) * target_val + interpolation_weight * subject_val,
                        ctrl_lower, ctrl_upper
                    )
                    
                    target_vel = aligned_target_velocities[idx, joint]
                    subject_vel = aligned_subject_velocities[idx, joint]
                    control_velocities[i] = (1 - interpolation_weight) * target_vel + interpolation_weight * subject_vel
                
                # Cubic Spline 보간
                cs_angle = CubicSpline(control_times, control_angles)
                cs_vel = CubicSpline(control_times, control_velocities)
                
                # 보간된 값 계산 및 경계 내로 제한
                for i in range(n_points):
                    lower_bound = min(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                    upper_bound = max(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                    
                    # 스플라인 값 계산 및 경계 내로 클리핑
                    spline_val = cs_angle(t[i])
                    interpolated_degrees[i, joint] = np.clip(spline_val, lower_bound, upper_bound)
                    
                    vel_val = cs_vel(t[i])
                    interpolated_velocities[i, joint] = vel_val
        
        # 2단계: 모델 기반 관절 간 상호작용 적용
        with torch.no_grad():
            # 긴 시퀀스는 세그먼트로 나누어 처리
            segments = []
            segment_size = 100
            
            for i in range(0, n_points, segment_size):
                # 세그먼트 추출 및 처리
                segment = interpolated_degrees[i:i+segment_size]
                if len(segment) == 0:
                    continue
                    
                # 모델 입력 준비 및 처리
                segment_tensor = torch.FloatTensor(segment).unsqueeze(0).to(self.device)
                joint_interactions = self.model(segment_tensor).squeeze(0).cpu().numpy()
                
                # 보정된 세그먼트 저장
                segments.append(joint_interactions)
            
            # 세그먼트 결합 및 모델 출력 처리
            if segments:
                model_output = np.vstack(segments)
                if len(model_output) > len(interpolated_degrees):
                    model_output = model_output[:len(interpolated_degrees)]
                
                # 모델 출력과 기본 보간 결과 블렌딩 - 항상 두 궤적 사이에 있도록 보장
                correction_strength = 0.25  # 호 보간에서는 약간 약한 보정 적용
                for joint in range(4):
                    for i in range(len(interpolated_degrees)):
                        # 경계 계산
                        lower_bound = min(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                        upper_bound = max(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                        
                        # 모델 출력을 경계 내로 제한
                        bounded_model_output = np.clip(model_output[i, joint], lower_bound, upper_bound)
                        
                        # 가중 평균 적용
                        original_val = interpolated_degrees[i, joint]
                        interpolated_degrees[i, joint] = (1 - correction_strength) * original_val + correction_strength * bounded_model_output
                        
                        # 최종 안전 점검
                        interpolated_degrees[i, joint] = np.clip(interpolated_degrees[i, joint], lower_bound, upper_bound)
            
        # 3단계: 각속도 재계산
        for joint in range(4):
            interpolated_velocities[:, joint] = np.gradient(interpolated_degrees[:, joint])

        return interpolated_degrees, interpolated_velocities

    def model_based_interpolate_circle(self, target, subject):
        """관절 관계를 고려한 모델 기반 원형 보간"""
        from scipy.spatial.transform import Rotation as R
        
        aligned_target, aligned_subject = self.normalize_time(target, subject)
        
        aligned_target_angles = aligned_target[:, :4]
        aligned_target_velocities = aligned_target[:, 4:]
        aligned_subject_angles = aligned_subject[:, :4]
        aligned_subject_velocities = aligned_subject[:, 4:]

        n_points = len(aligned_target_angles)
        interpolated_degrees = np.zeros_like(aligned_target_angles)
        interpolated_velocities = np.zeros_like(aligned_target_velocities)

        t = np.linspace(0, 1, n_points)
        weights = t * t * (3 - 2 * t)  # smooth step function
        
        # 1단계: 기본 쿼터니언 보간 수행
        for i in range(n_points):
            # 타겟과 서브젝트의 각도를 라디안으로 변환
            target_rad = np.radians(aligned_target_angles[i])
            subject_rad = np.radians(aligned_subject_angles[i])
            
            # 각도를 회전 객체로 변환 (처음 3개 관절만)
            q_target = R.from_euler('xyz', target_rad[:3])
            q_subject = R.from_euler('xyz', subject_rad[:3])
            
            # 쿼터니언 값 추출
            q_target_arr = q_target.as_quat()
            q_subject_arr = q_subject.as_quat()
            
            # SLERP 직접 구현
            dot = np.sum(q_target_arr * q_subject_arr)
            # 최단 경로 보장
            if dot < 0:
                q_subject_arr = -q_subject_arr
                dot = -dot
                
            # 각도가 매우 작은 경우 선형 보간
            if dot > 0.9995:
                result = q_target_arr + weights[i] * (q_subject_arr - q_target_arr)
                result = result / np.linalg.norm(result)
            else:
                theta_0 = np.arccos(dot)
                sin_theta_0 = np.sin(theta_0)
                
                theta = theta_0 * weights[i]
                sin_theta = np.sin(theta)
                
                s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
                s1 = sin_theta / sin_theta_0
                
                result = s0 * q_target_arr + s1 * q_subject_arr
                
            # 보간된 쿼터니언을 회전 객체로 변환
            q_interp = R.from_quat(result)
            
            # 오일러 각도로 변환
            euler_angles = q_interp.as_euler('xyz', degrees=True)
            interpolated_degrees[i, :3] = euler_angles
            
            # 4번째 조인트는 선형 보간 - 경계 확인
            target_val = aligned_target_angles[i, 3]
            subject_val = aligned_subject_angles[i, 3]
            lower_bound = min(target_val, subject_val)
            upper_bound = max(target_val, subject_val)
            
            # 선형 보간 후 경계 내로 제한
            interp_val = (1 - weights[i]) * target_val + weights[i] * subject_val
            interpolated_degrees[i, 3] = np.clip(interp_val, lower_bound, upper_bound)
            
            # 각속도 보간
            for j in range(4):
                v0 = aligned_target_velocities[i, j]
                v1 = aligned_subject_velocities[i, j]
                w = weights[i]
                
                # Hermite 보간
                h00 = 2*w**3 - 3*w**2 + 1
                h10 = w**3 - 2*w**2 + w
                h01 = -2*w**3 + 3*w**2
                h11 = w**3 - w**2
                
                interpolated_velocities[i, j] = h00*v0 + h10*0 + h01*v1 + h11*0
                
            # 오일러 각도도 경계 내에 있는지 확인
            for j in range(3):  # 첫 3개 관절 (쿼터니언으로 처리된)
                lower_bound = min(aligned_target_angles[i, j], aligned_subject_angles[i, j])
                upper_bound = max(aligned_target_angles[i, j], aligned_subject_angles[i, j])
                
                # 결과가 경계를 벗어났다면, 경계 내로 클리핑
                if interpolated_degrees[i, j] < lower_bound or interpolated_degrees[i, j] > upper_bound:
                    # 클리핑
                    interpolated_degrees[i, j] = np.clip(interpolated_degrees[i, j], lower_bound, upper_bound)

        # 2단계: 모델 기반 관절 간 상호작용 적용
        with torch.no_grad():
            # 긴 시퀀스는 세그먼트로 나누어 처리
            segments = []
            segment_size = 100
            
            for i in range(0, n_points, segment_size):
                # 세그먼트 추출 및 처리
                segment = interpolated_degrees[i:i+segment_size]
                if len(segment) == 0:
                    continue
                    
                # 모델 입력 준비 및 처리
                segment_tensor = torch.FloatTensor(segment).unsqueeze(0).to(self.device)
                joint_interactions = self.model(segment_tensor).squeeze(0).cpu().numpy()
                
                # 보정된 세그먼트 저장
                segments.append(joint_interactions)
            
            # 세그먼트 결합 및 모델 출력 처리
            if segments:
                model_output = np.vstack(segments)
                if len(model_output) > len(interpolated_degrees):
                    model_output = model_output[:len(interpolated_degrees)]
                
                # 모델 출력과 기본 보간 결과 블렌딩 - 항상 경계 내에 있도록 보장
                correction_strength = 0.2  # 원형 보간에서는 더 약한 보정 적용
                for joint in range(4):
                    for i in range(len(interpolated_degrees)):
                        # 경계 계산
                        lower_bound = min(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                        upper_bound = max(aligned_target_angles[i, joint], aligned_subject_angles[i, joint])
                        
                        # 모델 출력을 경계 내로 제한
                        bounded_model_output = np.clip(model_output[i, joint], lower_bound, upper_bound)
                        
                        # 가중 평균 적용
                        original_val = interpolated_degrees[i, joint]
                        interpolated_degrees[i, joint] = (1 - correction_strength) * original_val + correction_strength * bounded_model_output
                        
                        # 최종 안전 점검
                        interpolated_degrees[i, joint] = np.clip(interpolated_degrees[i, joint], lower_bound, upper_bound)
        
        # 3단계: 각속도 재계산
        for joint in range(4):
            interpolated_velocities[:, joint] = np.gradient(interpolated_degrees[:, joint])

        return interpolated_degrees, interpolated_velocities
    
    def interpolate_trajectory(self, target_df, user_df, trajectory_type):
        """궤적 타입에 따른 모델 기반 보간 수행"""        
        # 각속도 계산 및 DataFrame 생성
        target_with_vel = target_df.copy()
        user_with_vel = user_df.copy()
        
        # 각속도 계산 추가
        for df in [target_with_vel, user_with_vel]:
            df['degsec1'] = np.gradient(df['deg1'])
            df['degsec2'] = np.gradient(df['deg2'])
            df['degsec3'] = np.gradient(df['deg3'])
            df['degsec4'] = np.gradient(df['deg4'])
        
        # 스무딩 적용
        target_smoothed = self.smooth_data(target_with_vel)
        user_smoothed = self.smooth_data(user_with_vel)

        # 관절 각도와 각속도 데이터 준비
        target_angles = target_smoothed[['deg1', 'deg2', 'deg3', 'deg4']].values
        target_velocities = target_smoothed[['degsec1', 'degsec2', 'degsec3', 'degsec4']].values
        
        user_angles = user_smoothed[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_velocities = user_smoothed[['degsec1', 'degsec2', 'degsec3', 'degsec4']].values
        
        target_data = np.column_stack([target_angles, target_velocities])
        user_data = np.column_stack([user_angles, user_velocities])
        
        # 보간 방법 선택 및 적용 - 궤적 유형에 따라 다른 보간 방법 사용
        if 'clock' in trajectory_type.lower() or 'counter' in trajectory_type.lower():
            print("원형 보간 사용 중 (모델 기반)")
            aligned_degrees, aligned_velocities = self.model_based_interpolate_circle(target_data, user_data)
        elif 'v_' in trajectory_type.lower() or 'h_' in trajectory_type.lower():
            print("호 보간 사용 중 (모델 기반)")
            aligned_degrees, aligned_velocities = self.model_based_interpolate_arc(target_data, user_data)
        else:
            print("선형 보간 사용 중 (모델 기반)")
            aligned_degrees, aligned_velocities = self.model_based_interpolate_line(target_data, user_data)
        
        # 보간된 관절 각도로부터 end-effector 위치 계산
        endeffector_degrees = aligned_degrees.copy()
        endeffector_degrees[:, 1] -= 90  # 관절 변환 적용
        endeffector_degrees[:, 3] -= 90

        aligned_points = np.array([calculate_end_effector_position(deg) for deg in endeffector_degrees])
        aligned_points = aligned_points * 1000  # 밀리미터 단위로 변환

        # 결과 데이터프레임 생성
        generated_df = pd.DataFrame(
            np.column_stack([aligned_points, aligned_degrees]),
            columns=['x_end', 'y_end', 'z_end', 'deg1', 'deg2', 'deg3', 'deg4']
        )
        
        # 최종 확인: 모든 관절 값이 타겟과 사용자 궤적 사이에 있는지 확인
        for joint in ['deg1', 'deg2', 'deg3', 'deg4']:
            for i in range(len(generated_df)):
                if i < len(target_df) and i < len(user_df):
                    lower_bound = min(target_df[joint].iloc[i], user_df[joint].iloc[i])
                    upper_bound = max(target_df[joint].iloc[i], user_df[joint].iloc[i])
                    
                    # 최종 결과가 경계를 벗어났는지 확인
                    if generated_df[joint].iloc[i] < lower_bound or generated_df[joint].iloc[i] > upper_bound:
                        # 범위를 벗어나면 경계로 조정
                        generated_df.at[i, joint] = np.clip(generated_df[joint].iloc[i], lower_bound, upper_bound)
                        
                        # 엔드이펙터 위치 재계산
                        adjusted_degrees = generated_df.loc[i, ['deg1', 'deg2', 'deg3', 'deg4']].values
                        adjusted_effector_degrees = adjusted_degrees.copy()
                        adjusted_effector_degrees[1] -= 90
                        adjusted_effector_degrees[3] -= 90
                        
                        new_position = calculate_end_effector_position(adjusted_effector_degrees) * 1000
                        generated_df.at[i, 'x_end'] = new_position[0]
                        generated_df.at[i, 'y_end'] = new_position[1]
                        generated_df.at[i, 'z_end'] = new_position[2]
        
        return generated_df

    def visualize_trajectories(self, target_df, user_df, generated_df, trajectory_type, generation_number=1):
        """타겟과 사용자 궤적과 생성된 궤적을 시각화"""
        # 관절 각도 데이터 준비
        target_degrees = target_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_degrees = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        generated_degrees = generated_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        
        # End-effector 위치 계산 - 시각화를 위한 변환 적용
        target_degrees_adj = target_degrees.copy()
        target_degrees_adj[:, 1] -= 90
        target_degrees_adj[:, 3] -= 90
        
        user_degrees_adj = user_degrees.copy()
        user_degrees_adj[:, 1] -= 90
        user_degrees_adj[:, 3] -= 90
        
        target_ends = np.array([calculate_end_effector_position(deg) for deg in target_degrees_adj]) * 1000
        user_ends = np.array([calculate_end_effector_position(deg) for deg in user_degrees_adj]) * 1000

        # 시각화
        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(2, 3, figure=fig)
        
        # 3D end-effector 궤적 시각화
        ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
        ax_3d.plot(target_ends[:, 0], target_ends[:, 1], target_ends[:, 2], 
                'b-', label='target trajectory')
        ax_3d.plot(user_ends[:, 0], user_ends[:, 1], user_ends[:, 2],
                'r-', label='user trajectory')
        ax_3d.plot(generated_df['x_end'], generated_df['y_end'], generated_df['z_end'], 
                'g-', label='interpolated trajectory')
        
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title('End-effector Trajectory')
        ax_3d.legend()

        # 관절별 그래프
        target_time = np.arange(len(target_degrees))
        user_time = np.arange(len(user_degrees))
        aligned_time = np.arange(len(generated_degrees))
        joint_titles = ['deg1', 'deg2', 'deg3', '4']
        for idx, joint in enumerate(['deg1', 'deg2', 'deg3', 'deg4']):
            row = idx // 2
            col = (idx % 2) + 1
            
            ax = fig.add_subplot(gs[row, col])
            
            ax.plot(target_time, target_df[joint], 'b--', label='target')
            ax.plot(user_time, user_df[joint], 'r:', label='user')
            ax.plot(aligned_time, generated_df[joint], 'g-', label='interpolate')
            
            ax.set_title(joint_titles[idx])
            ax.set_xlabel('timestampe')
            ax.set_ylabel('deg')
            ax.grid(True)
            ax.legend()

            # 영역 표시 - 두 궤적 사이의 허용 범위
            for i in range(min(len(target_time), len(user_time))):
                if i < len(target_time) and i < len(user_time):
                    lower = min(target_df[joint].iloc[i], user_df[joint].iloc[i])
                    upper = max(target_df[joint].iloc[i], user_df[joint].iloc[i])
                    ax.fill_between([i, i+1], [lower, lower], [upper, upper], color='gray', alpha=0.2)
            
            # 경계를 벗어난 지점 확인 및 표시
            for i in range(len(generated_df)):
                if i < len(target_df) and i < len(user_df):
                    lower = min(target_df[joint].iloc[i], user_df[joint].iloc[i])
                    upper = max(target_df[joint].iloc[i], user_df[joint].iloc[i])
                    val = generated_df[joint].iloc[i]
                    
                    # 범위를 벗어났는지 확인
                    if val < lower or val > upper:
                        ax.plot(i, val, 'rx', markersize=8)  # 경계를 벗어난 점 표시

        fig.suptitle(f'Trajectory_type: {trajectory_type}', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

        # 생성된 궤적 저장
        self.save_generated_trajectory(generated_df, trajectory_type, generation_number)
        
        return generated_df
    
    def save_generated_trajectory(self, generated_df, classification_result, generation_number=1):
        """생성된 궤적을 지정된 형식으로 저장"""
        generation_dir = os.path.join(os.getcwd(), "generation_trajectory")
        os.makedirs(generation_dir, exist_ok=True)
        
        filename = f"generation_trajectory_{classification_result}_{generation_number}.txt"
        generation_path = os.path.join(generation_dir, filename)
        
        # 데이터프레임 생성 및 형식 맞추기
        num_points = len(generated_df)
        full_df = pd.DataFrame(index=range(num_points))
        
        # 기본 필드 설정
        full_df['r'] = 'm'
        full_df['sequence'] = range(num_points)
        full_df['timestamp'] = [i * 10 for i in range(num_points)]  # 10ms 간격
        
        # 각도, 엔드이펙터터의 데이터 설정
        full_df['deg'] = (generated_df['deg1'].round(3).astype(str) + '/' + 
                        generated_df['deg2'].round(3).astype(str) + '/' +
                        generated_df['deg3'].round(3).astype(str) + '/' +
                        generated_df['deg4'].round(3).astype(str))

        full_df['endpoint'] = (generated_df['x_end'].round(3).astype(str) + '/' + 
                            generated_df['y_end'].round(3).astype(str) + '/' + 
                            generated_df['z_end'].round(3).astype(str))
        
        # 실제 파일 형식에 맞게 열 순서 및 포맷 조정
        full_df = full_df[['r', 'sequence', 'timestamp', 'deg', 'endpoint']]
        
        full_df.to_csv(generation_path, index=False)
        print(f"\n생성된 궤적 저장 완료: {generation_path}")

        return generation_path
        
    def analyze_joint_relationships(self):
        """관절 간 상관관계 분석 및 시각화"""
        # 모델의 가중치 분석
        # 첫 번째 joint attention layer의 가중치 추출
        with torch.no_grad():
            # 관절 간 관계 가시화를 위한 간단한 입력 데이터 생성
            dummy_input = torch.eye(4).unsqueeze(0).to(self.device)
            dummy_input = self.model.joint_embedding(dummy_input)
            
            # Joint Attention의 query, key, value 가중치 추출
            attentions = []
            for layer in self.model.joint_attention_layers:
                Q = layer.query(dummy_input)
                K = layer.key(dummy_input)
                scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(layer.d_model).float())
                attention = torch.softmax(scores, dim=-1)
                attentions.append(attention.squeeze(0).cpu().numpy())
        
        # 관절 간 관계 시각화
        fig, axes = plt.subplots(1, len(attentions), figsize=(len(attentions) * 5, 5))
        joint_names = ['deg1', 'deg2', 'deg3', 'deg4']
        
        if len(attentions) == 1:
            axes = [axes]
            
        for i, attn in enumerate(attentions):
            ax = axes[i]
            im = ax.imshow(attn, cmap='viridis')
            ax.set_title(f'Joint Attention Layer {i+1}')
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(joint_names)
            ax.set_yticklabels(joint_names)
            
            # 각 셀에 값 표시
            for j in range(4):
                for k in range(4):
                    text = ax.text(k, j, f'{attn[j, k]:.2f}',
                                  ha="center", va="center", color="w" if attn[j, k] > 0.5 else "black")
            
            fig.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.suptitle('Interarticular Correlation Analysis', fontsize=16)
        plt.subplots_adjust(top=0.85)
        plt.show()
        
        # 관절 간 상관관계 설명
        print("\n관절 간 상관관계 분석 결과:")
        print("----------------------------")
        print("값이 1.0에 가까울수록 관절 간 상관관계가 강함을 의미합니다.")
        print("이는 하나의 관절이 움직일 때 다른 관절이 조화롭게 움직이는 경향을 나타냅니다.")
        
        # 가장 강한 관계 찾기
        avg_attention = np.mean(np.array(attentions), axis=0)
        np.fill_diagonal(avg_attention, 0)  # 대각선(자기 자신과의 관계)은 무시
        
        strongest_pairs = []
        for i in range(4):
            for j in range(i+1, 4):
                strongest_pairs.append((i, j, avg_attention[i, j]))
        
        # 상관관계가 강한 순서로 정렬
        strongest_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print("\n가장 강한 관절 간 상관관계 (상위 3개):")
        for i, (joint1, joint2, strength) in enumerate(strongest_pairs[:3]):
            print(f"{i+1}. {joint_names[joint1]} ↔ {joint_names[joint2]}: {strength:.4f}")

def main():
    """메인 함수 - 데이터셋 학습 및 궤적 생성 실행"""
    base_dir = os.path.join(os.getcwd(), "data")
    model_path = os.path.join(os.getcwd(), "models", "joint_relationship_model.pth")
    
    try:
        # 객체 초기화
        print("\n궤적 분석기 및 생성기 초기화 중...")
        analyzer = TrajectoryAnalyzer(
            classification_model="best_classification_model.pth",
            base_dir=base_dir
        )
        generator = ModelBasedTrajectoryGenerator(analyzer, model_path)
        
        # 사용자 궤적 파일 로드
        print("\n사용자 궤적 로드 및 분류 중...")
        non_golden_dir = os.path.join(base_dir, "non_golden_sample")
        non_golden_files = [f for f in os.listdir(non_golden_dir) if f.endswith('.txt')]
        
        if not non_golden_files:
            raise ValueError("non_golden_sample 디렉토리에 궤적 파일이 없습니다.")
        
        selected_file = random.choice(non_golden_files)
        print(f"선택된 사용자 궤적: {selected_file}")
        
        file_path = os.path.join(non_golden_dir, selected_file)

        # 궤적 로드 및 분류
        user_trajectory, trajectory_type = analyzer.load_user_trajectory(file_path)
        target_trajectory, _ = analyzer.load_target_trajectory(trajectory_type)
        
        # 관절 간 상관관계 분석
        print("\n관절 간 상관관계 분석 중...")
        generator.analyze_joint_relationships()
        
        # 모델 기반 궤적 생성
        print("\n모델 기반 궤적 생성 중...")
        generated_df = generator.interpolate_trajectory(
            target_df=target_trajectory,
            user_df=user_trajectory,
            trajectory_type=trajectory_type
        )

        # 시각화 및 저장
        print("\n궤적 시각화 및 저장 중...")
        generator.visualize_trajectories(
            target_df=target_trajectory,
            user_df=user_trajectory,
            generated_df=generated_df,
            trajectory_type=trajectory_type
        )
        
        print("\n처리 완료!")
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        print("데이터 디렉토리 구조와 모델 경로를 확인하세요.")

if __name__ == "__main__":
    main()