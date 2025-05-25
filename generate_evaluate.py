import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.interpolate import UnivariateSpline, CubicSpline
from scipy.signal import savgol_filter
from utils import calculate_end_effector_position
from analyzer import TrajectoryAnalyzer
from endeffector_model import TrajectoryTransformer
from evaluate import *

class EndeffectorGenerator:
    def __init__(self, analyzer, model_path="best_trajectory_transformer.pth"):
        """ 트랜스포머 기반 엔드이펙터 궤적 생성기 """
        self.analyzer = analyzer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 관절 제한 설정
        self.joint_limits = {
            0: (-10, 110),
            1: (0, 150),     
            2: (0, 150),     
            3: (-90, 90)  
        }

        # 평가 결과에 따른 가중치 적용용
        self.grade_to_weight = {
            1: 0.2,  # 1등급 
            2: 0.4,  # 2등급
            3: 0.6,  # 3등급
            4: 0.8   # 4등급 
        }
        # 모델 불러오기
        self.transformer = self.load_transformer_model(model_path)
        
    def load_transformer_model(self, model_path):
        """학습된 트랜스포머 모델 불러오기"""
        try:
            # 모델 초기화
            transformer = TrajectoryTransformer(
                            input_dim=7,
                            d_model=64,    
                            nhead=2,         
                            num_layers=2    
                        )
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                transformer.load_state_dict(checkpoint['model_state_dict'])
                print(f"모델 가중치 로드 완료: epoch {checkpoint.get('epoch', 'unknown')}, loss {checkpoint.get('loss', 'unknown')}")
            else:
                transformer.load_state_dict(checkpoint)
                print(f"모델 가중치 로드 완료")
            
            # 평가 모드로 설정
            transformer.to(self.device)
            transformer.eval()
            
            print(f"트랜스포머 모델 로드 성공: {model_path}")
            return transformer
            
        except Exception as e:
            print(f"모델 로드 중 에러 발생: {str(e)}")
            return None

    def smooth_data(self, data, smoothing_factor=1.0, degree=3):
        """궤적 스플라인 스무딩"""
        # 원본 데이터 복사
        smoothed_df = data.copy()

        # 관절 각도 추출
        degrees = data[['deg1', 'deg2', 'deg3', 'deg4']].values

        # 엔드이펙터 위치 계산
        endpoints = np.array([calculate_end_effector_position(deg) for deg in degrees]) * 1000

        # 엔드이펙터 위치를 x, y, z로 분리
        x = endpoints[:, 0]
        y = endpoints[:, 1]
        z = endpoints[:, 2]
        t = np.arange(len(data))

        # 스플라인 스무딩 적용
        s = smoothing_factor * len(data)
        spline_x = UnivariateSpline(t, x, k=degree, s=s)
        spline_y = UnivariateSpline(t, y, k=degree, s=s)
        spline_z = UnivariateSpline(t, z, k=degree, s=s)

        # 스무딩된 엔드이펙터 위치
        smoothed_endpoints = np.column_stack([spline_x(t), spline_y(t), spline_z(t)])

        # 결과를 DataFrame에 저장
        smoothed_df['x'] = smoothed_endpoints[:, 0]
        smoothed_df['y'] = smoothed_endpoints[:, 1]
        smoothed_df['z'] = smoothed_endpoints[:, 2]

        return smoothed_df
    
    def normalize_time(self, target_data, user_data, num_points=100):
        """
        시간 정규화 및 DTW 적용 - 간소화된 버전
        불필요한 분석 없이 핵심 기능에만 집중
        """
        print(f"DTW 정렬 및 리샘플링 시작: {num_points} 포인트로 변환")
        
        try:
            # 1단계: 스무딩 적용
            target_smoothed = self.smooth_data(target_data, smoothing_factor=0.5, degree=3)
            user_smoothed = self.smooth_data(user_data, smoothing_factor=0.5, degree=3)
            
            # 2단계: 관절 각도 추출
            target_degrees = target_smoothed[['deg1', 'deg2', 'deg3', 'deg4']].values
            user_degrees = user_smoothed[['deg1', 'deg2', 'deg3', 'deg4']].values

            # 3단계: 엔드이펙터 위치 계산 (DTW 정렬용)
            target_endpoint = np.array([calculate_end_effector_position(deg) for deg in target_degrees]) * 1000
            user_endpoint = np.array([calculate_end_effector_position(deg) for deg in user_degrees]) * 1000

            # 4단계: DTW 정렬
            _, path = fastdtw(target_endpoint, user_endpoint, dist=euclidean)
            path = np.array(path)    
            print(f"DTW 정렬 완료 - 매칭 포인트: {len(path)}개")

            # 5단계: 정렬된 데이터 생성
            # 엔드이펙터 위치 정렬
            aligned_target_endpoint = np.array([target_endpoint[i] for i in path[:, 0]])
            aligned_user_endpoint = np.array([user_endpoint[j] for j in path[:, 1]])
            
            # 관절 각도 정렬 (같은 시간 대응 관계 사용)
            aligned_target_degrees = np.array([target_degrees[i] for i in path[:, 0]])
            aligned_user_degrees = np.array([user_degrees[j] for j in path[:, 1]])

            # 6단계: 7차원 완전 궤적 데이터 생성
            # [x, y, z, deg1, deg2, deg3, deg4] 형태로 결합
            target_complete = np.concatenate([aligned_target_endpoint, aligned_target_degrees], axis=1)
            user_complete = np.concatenate([aligned_user_endpoint, aligned_user_degrees], axis=1)
            print(f"7차원 궤적 생성 완료 - Target: {target_complete.shape}, User: {user_complete.shape}")

            # 7단계: 리샘플링 (목표 길이로 조정)
            def simple_resample(data, target_length):
                """간단하고 안정적인 리샘플링"""
                if len(data) <= 1:
                    # 데이터가 너무 적으면 복제
                    return np.tile(data[0] if len(data) == 1 else np.zeros(7), (target_length, 1))
                
                # 시간축 정의
                t_old = np.linspace(0, 1, len(data))
                t_new = np.linspace(0, 1, target_length)
                
                # 각 차원별로 보간
                result = np.zeros((target_length, data.shape[1]))
                for i in range(data.shape[1]):
                    try:
                        # 3차 스플라인 시도
                        spline = CubicSpline(t_old, data[:, i], bc_type='clamped')
                        result[:, i] = spline(t_new)
                    except:
                        # 실패하면 선형 보간
                        result[:, i] = np.interp(t_new, t_old, data[:, i])
                
                return result

            # 실제 리샘플링 실행
            resampled_target = simple_resample(target_complete, num_points)
            resampled_user = simple_resample(user_complete, num_points)
            
            print(f"리샘플링 완료 - 최종 크기: {resampled_target.shape}")

            # 8단계: 기본적인 데이터 정리 (NaN 제거만)
            # 복잡한 분석 없이 기본적인 문제만 해결
            def clean_data(data):
                """NaN 값만 간단히 제거"""
                if np.any(np.isnan(data)):
                    print("NaN 값 발견, 선형 보간으로 수정")
                    for i in range(data.shape[1]):
                        col = data[:, i]
                        if np.any(np.isnan(col)):
                            # 유효한 값들로 보간
                            valid_mask = ~np.isnan(col)
                            if np.any(valid_mask):
                                data[:, i] = np.interp(
                                    np.arange(len(col)),
                                    np.where(valid_mask)[0],
                                    col[valid_mask]
                                )
                            else:
                                data[:, i] = 0  # 모든 값이 NaN이면 0으로
                return data

            resampled_target = clean_data(resampled_target)
            resampled_user = clean_data(resampled_user)

            print("데이터 정리 완료, 7차원 궤적 반환")
            return resampled_target, resampled_user
            
        except Exception as e:
            print(f"처리 중 오류: {str(e)}")
            return None
    
    # def prepare_trajectory_input(self, target_df, user_df, max_seq_length=100):
    #     """트랜스포머 모델 입력을 위한 궤적 데이터 준비 (위치/각도 분리 정규화)"""
        
    #     def process_trajectory_inference(df):
    #         degrees = df[['deg1', 'deg2', 'deg3', 'deg4']].values
    #         endpoints = np.array([calculate_end_effector_position(deg) for deg in degrees]) * 1000
    #         combined = np.concatenate([endpoints, degrees], axis=1) 
            
    #         t_orig = np.linspace(0, 1, len(combined))
    #         t_new = np.linspace(0, 1, max_seq_length)
            
    #         resampled = np.zeros((max_seq_length, combined.shape[1]))
    #         for i in range(combined.shape[1]):
    #             spline = CubicSpline(t_orig, combined[:, i])
    #             resampled[:, i] = spline(t_new)

    #         mean = np.mean(resampled, axis=0)
    #         std = np.std(resampled, axis=0) + 1e-8
    #         normalized = (resampled - mean) / std
            
    #         return normalized, mean, std

    #     target_normalized, target_mean, target_std = process_trajectory_inference(target_df)
    #     user_normalized, user_mean, user_std = process_trajectory_inference(user_df)

    #     alpha = 0.5 
    #     interpolated_gt = alpha * user_normalized + (1 - alpha) * target_normalized
        
    #     normalization_params = {
    #         'mean': user_mean,
    #         'std': user_std,
    #         'max_seq_length': max_seq_length
    #     }
        
    #     # 텐서 변환
    #     target_tensor = torch.FloatTensor(target_normalized).unsqueeze(0).to(self.device)
    #     user_tensor = torch.FloatTensor(user_normalized).unsqueeze(0).to(self.device)
    #     interpolated_tensor = torch.FloatTensor(interpolated_gt).unsqueeze(0).to(self.device)
        
    #     return user_tensor, target_tensor, interpolated_tensor, normalization_params

    def generate_with_fixed_weight(self, user_tensor, target_tensor, fixed_weight):
        """ 지정된 가중치를 사용하여 궤적 생성 """
        # 배치 크기,시퀀스 길이, 입력 차원 불러오기
        batch_size, seq_len, input_dim = user_tensor.size()
        
        # 사용자와 타겟 궤적을 각각 임베딩 처리
        # user_emb = self.transformer.positional_encoding(
        #     self.transformer.user_embedding(user_tensor)
        # )
        # target_emb = self.transformer.positional_encoding(
        #     self.transformer.target_embedding(target_tensor)
        # )
        
        # 수정
        user_emb = self.transformer.user_encoder(
            self.transformer.positional_encoding(
                self.transformer.user_embedding(user_tensor)
            )
        )
        target_emb = self.transformer.target_encoder(
            self.transformer.positional_encoding(
                self.transformer.target_embedding(target_tensor)
            )
        )

        # 평가 결과를 기반으로 한 가중치 조정
        weight_tensor = torch.tensor([fixed_weight], device=user_tensor.device).float()
        weight_tensor = weight_tensor.unsqueeze(0).unsqueeze(-1) 

        memory = weight_tensor * user_emb + (1 - weight_tensor) * target_emb

        if hasattr(self.transformer, 'temporal_encoder'):
            memory = memory + self.transformer.temporal_encoder(memory)
        
        # 자기회귀적 생성
        generated = torch.zeros(batch_size, seq_len, input_dim).to(user_tensor.device)
        
        generated[:, 0, :] = (
            weight_tensor.squeeze() * user_tensor[:, 0, :] + 
            (1 - weight_tensor.squeeze()) * target_tensor[:, 0, :]
        )

        for t in range(1, seq_len):
            current_input = generated[:, :t, :]

            tgt_emb = self.transformer.output_embedding(current_input)
            tgt_emb = self.transformer.positional_encoding(tgt_emb)

            tgt_mask = self.transformer._generate_square_subsequent_mask(t).to(user_tensor.device)

            decoder_output = self.transformer.decoder(tgt_emb, memory[:, :t, :], tgt_mask=tgt_mask)
            next_step = self.transformer.output_layer(decoder_output[:, -1:, :])
            
            # 예측된 값 저장
            generated[:, t, :] = next_step.squeeze(1)
        
        return generated
    
    def prepare_trajectory_input(self, target_df, user_df, max_seq_length=100):
        """
        완전한 DTW 정렬 및 통합 정규화를 사용한 궤적 데이터 준비
        
        이제 normalize_time이 완전한 7차원 데이터를 반환하므로,
        복잡한 관절 각도 추정 과정 없이 직접 사용할 수 있습니다.
        """
        print("=== 완전한 궤적 전처리 시작 ===")
        print(f"입력 데이터 크기 - Target: {len(target_df)}, User: {len(user_df)}")
        
        # 1단계: DTW 정렬과 완전한 궤적 데이터 생성
        # 이제 normalize_time이 엔드이펙터 위치 + 관절 각도의 완전한 7차원 데이터를 반환합니다
        print("DTW 정렬 및 완전한 궤적 정규화 적용 중...")
        
        target_complete, user_complete = self.normalize_time(
            target_df, user_df, num_points=max_seq_length
        )
        
        print(f"DTW 정렬 완료:")
        print(f"  - Target 완전 궤적: {target_complete.shape}")
        print(f"  - User 완전 궤적: {user_complete.shape}")
        
        # 2단계: 데이터 형식 검증
        # 7차원 데이터가 올바르게 구성되었는지 확인합니다
        expected_shape = (max_seq_length, 7)  # (시퀀스 길이, 7차원)
        
        if target_complete.shape != expected_shape or user_complete.shape != expected_shape:
            print(f"⚠️ 경고: 예상 형태 {expected_shape}와 다릅니다")
            print(f"실제 형태 - Target: {target_complete.shape}, User: {user_complete.shape}")
        
        # 각 차원의 의미를 명확히 하기 위한 검증
        print("데이터 구성 검증:")
        print(f"  - 위치 차원 (0-2): X, Y, Z 엔드이펙터 좌표")
        print(f"  - 각도 차원 (3-6): deg1, deg2, deg3, deg4 관절 각도")
        
        # 위치 데이터 범위 확인 (mm 단위여야 함)
        pos_ranges = {
            'X': (target_complete[:, 0].min(), target_complete[:, 0].max()),
            'Y': (target_complete[:, 1].min(), target_complete[:, 1].max()), 
            'Z': (target_complete[:, 2].min(), target_complete[:, 2].max())
        }
        
        print("위치 데이터 범위 (mm):")
        for axis, (min_val, max_val) in pos_ranges.items():
            print(f"  - {axis}: {min_val:.2f} ~ {max_val:.2f}")
        
        # 각도 데이터 범위 확인 (degree 단위여야 함)
        angle_ranges = {}
        for i in range(4):
            joint_name = f"deg{i+1}"
            min_val = target_complete[:, 3+i].min()
            max_val = target_complete[:, 3+i].max()
            angle_ranges[joint_name] = (min_val, max_val)
        
        print("각도 데이터 범위 (degree):")
        for joint, (min_val, max_val) in angle_ranges.items():
            print(f"  - {joint}: {min_val:.2f} ~ {max_val:.2f}")

        # 3단계: 통합된 정규화 적용
        # 훈련 시와 동일하게 두 궤적을 함께 고려한 정규화를 수행합니다
        print("통합 정규화 수행 중...")
        
        # 전체 데이터를 하나로 합쳐서 통합 통계 계산
        all_trajectory_data = np.concatenate([target_complete, user_complete], axis=0)
        
        # 통합된 평균과 표준편차 계산
        unified_mean = np.mean(all_trajectory_data, axis=0)
        unified_std = np.std(all_trajectory_data, axis=0) + 1e-8  # 0으로 나누기 방지
        
        print(f"통합 정규화 파라미터:")
        print(f"  - 평균: {unified_mean}")
        print(f"  - 표준편차: {unified_std}")
        
        # 정규화 적용
        target_normalized = (target_complete - unified_mean) / unified_std
        user_normalized = (user_complete - unified_mean) / unified_std
        
        print(f"정규화 완료:")
        print(f"  - Target 정규화: 평균={np.mean(target_normalized):.6f}, 표준편차={np.std(target_normalized):.6f}")
        print(f"  - User 정규화: 평균={np.mean(user_normalized):.6f}, 표준편차={np.std(user_normalized):.6f}")

        # 4단계: 훈련과 동일한 방식의 보간 기준 생성
        # 모델이 학습할 때와 같은 방식으로 ground truth를 만듭니다
        print("보간 기준 생성 중...")
        
        # 기본 알파 값 (추후 평가 등급에 따라 조정 가능)
        default_alpha = 0.5
        interpolated_gt = default_alpha * user_normalized + (1 - default_alpha) * target_normalized
        
        print(f"보간 완료 (알파={default_alpha}):")
        print(f"  - 보간 결과: 평균={np.mean(interpolated_gt):.6f}, 표준편차={np.std(interpolated_gt):.6f}")

        # 5단계: 정규화 파라미터와 메타데이터 저장
        # 역정규화와 추가 처리를 위한 모든 필요한 정보를 저장합니다
        normalization_params = {
            'unified_mean': unified_mean,
            'unified_std': unified_std,
            'max_seq_length': max_seq_length,
            'dtw_aligned': True,  # DTW 정렬이 적용되었음을 표시
            'data_format': '7d_complete',  # 7차원 완전 궤적 데이터임을 표시
            'target_raw': target_complete,  # 정규화 전 원본 데이터
            'user_raw': user_complete,
            'data_ranges': {
                'position': pos_ranges,
                'angles': angle_ranges
            }
        }
        
        print("메타데이터 저장 완료")

        # 6단계: PyTorch 텐서로 변환
        # GPU 연산을 위해 적절한 형태로 변환합니다
        print("PyTorch 텐서로 변환 중...")
        
        try:
            target_tensor = torch.FloatTensor(target_normalized).unsqueeze(0).to(self.device)
            user_tensor = torch.FloatTensor(user_normalized).unsqueeze(0).to(self.device)
            interpolated_tensor = torch.FloatTensor(interpolated_gt).unsqueeze(0).to(self.device)
            
            print(f"텐서 변환 완료:")
            print(f"  - Target: {target_tensor.shape} on {target_tensor.device}")
            print(f"  - User: {user_tensor.shape} on {user_tensor.device}")
            print(f"  - Interpolated: {interpolated_tensor.shape} on {interpolated_tensor.device}")
            
        except Exception as e:
            print(f"텐서 변환 중 오류 발생: {str(e)}")
            raise

        # 7단계: 최종 검증
        # 생성된 텐서들이 모델 입력으로 적합한지 확인합니다
        print("최종 검증 수행 중...")
        
        # 차원 검증
        expected_tensor_shape = (1, max_seq_length, 7)  # (배치, 시퀀스, 특성)
        
        shapes_correct = (
            target_tensor.shape == expected_tensor_shape and
            user_tensor.shape == expected_tensor_shape and
            interpolated_tensor.shape == expected_tensor_shape
        )
        
        return user_tensor, target_tensor, interpolated_tensor, normalization_params

    def denormalize_trajectory(self, generated_np, normalization_params):
        """
        통합된 정규화 기준을 사용한 단순하고 일관된 역정규화
        훈련과 추론 간의 일관성을 보장하기 위해 복잡한 가중치 혼합을 제거
        """
        print("통합 정규화 기준을 사용한 역정규화 시작...")
        
        # 통합된 정규화 파라미터 사용
        unified_mean = normalization_params['unified_mean']
        unified_std = normalization_params['unified_std']
        
        # 단순한 역정규화 적용
        # generated_np * std + mean 공식으로 원래 스케일 복원
        denormalized = generated_np * unified_std + unified_mean
        
        print(f"역정규화 완료 - 출력 형태: {denormalized.shape}")
        
        # 위치와 각도 분리
        # 처음 3차원: 엔드이펙터 위치 (mm)
        # 나머지 4차원: 관절 각도 (degree)
        generated_pos = denormalized[:, :3]  # 엔드이펙터 위치
        generated_deg = denormalized[:, 3:]  # 관절 각도
        
        print(f"위치 범위: X({generated_pos[:, 0].min():.2f}~{generated_pos[:, 0].max():.2f}), "
            f"Y({generated_pos[:, 1].min():.2f}~{generated_pos[:, 1].max():.2f}), "
            f"Z({generated_pos[:, 2].min():.2f}~{generated_pos[:, 2].max():.2f})")
        
        print(f"각도 범위: deg1({generated_deg[:, 0].min():.2f}~{generated_deg[:, 0].max():.2f}), "
            f"deg2({generated_deg[:, 1].min():.2f}~{generated_deg[:, 1].max():.2f}), "
            f"deg3({generated_deg[:, 2].min():.2f}~{generated_deg[:, 2].max():.2f}), "
            f"deg4({generated_deg[:, 3].min():.2f}~{generated_deg[:, 3].max():.2f})")
        
        return generated_pos, generated_deg
    
    # def denormalize_with_interpolation(self, generated_np, normalization_params, applied_weight):
    #     """
    #     평가 등급을 반영한 지능적 역정규화
    #     모델의 출력을 실제로 활용하면서도 사용자 수준에 맞는 스케일 조정
    #     """
        
    #     # 가중치에 따라 정규화 기준을 조합
    #     user_mean = normalization_params['user_mean']
    #     user_std = normalization_params['user_std']
    #     target_mean = normalization_params['target_mean']
    #     target_std = normalization_params['target_std']
        
    #     # 핵심: 모델의 출력을 실제로 사용!
    #     # 가중치에 따라 정규화 파라미터를 보간하여 적절한 스케일 결정
    #     interpolated_mean = applied_weight * user_mean + (1 - applied_weight) * target_mean
    #     interpolated_std = applied_weight * user_std + (1 - applied_weight) * target_std
        
    #     # 모델이 생성한 정규화된 궤적을 역정규화
    #     denormalized = generated_np * interpolated_std + interpolated_mean
        
    #     # 위치와 각도 분리
    #     generated_pos = denormalized[:, :3]  # 엔드이펙터 위치
    #     generated_deg = denormalized[:, 3:]  # 관절 각도
        
    #     return generated_pos, generated_deg

    def generate_trajectory(self, target_df, user_df, trajectory_type, weight=None):
        """트랜스포머 모델을 사용하여 새로운 궤적 생성"""
        
        # 모델이 제대로 로드되었는지 확인
        if self.transformer is None:
            print("트랜스포머 모델이 로드되지 않았습니다.")
            return None, None

        try:
            # 데이터 전처리
            # user_tensor, target_tensor, interpolated_tensor, norm_params = self.prepare_trajectory_input(
            #     target_df, user_df)
            user_tensor, target_tensor, interpolated_tensor, norm_params = self.prepare_trajectory_input(
                                                                            target_df, user_df
                                                                        )

            # 모델을 통한 궤적 생성
            with torch.no_grad(): 
                if weight is not None:
                    # 평가 결과에 따른 특정 가중치가 제공된 경우
                    print(f"지정된 가중치 {weight:.2f}로 궤적 생성 시도 중...")
                    
                    # 사용자 정의 가중치를 사용하여 궤적 생성
                    # 이 방법은 모델의 적응적 가중치 시스템을 우회합니다
                    generated = self.generate_with_fixed_weight(
                        user_tensor, target_tensor, weight
                    )
                    
                    print(f"지정된 가중치 {weight:.2f}가 정확히 적용됨")
                    predicted_weight = torch.tensor([weight])
                    
                else:
                    # 가중치가 지정되지 않은 경우 모델의 적응적 가중치 사용
                    print("모델의 적응적 가중치를 사용하여 궤적 생성 중...")
                    result = self.transformer(user_tensor, target_tensor, interpolated_gt=None)
                    
                    # 모델 출력 형태에 따른 처리
                    if isinstance(result, tuple):
                        generated, predicted_weight = result
                    else:
                        generated = result
                        predicted_weight = None
                    print("모델의 적응적 가중치 사용됨")

            generated_np = generated.squeeze(0).cpu().numpy()

            # 역정규화 과정
            generated_pos, generated_deg = self.denormalize_trajectory(
                                generated_np, norm_params
                            )


            # 물리적 제약 조건 적용
            for joint_idx, (min_angle, max_angle) in self.joint_limits.items():
                generated_deg[:, joint_idx] = np.clip(
                    generated_deg[:, joint_idx], min_angle, max_angle
                )

            # 결과 반환
            generated_df = pd.DataFrame({
                'x_end': generated_pos[:, 0],     
                'y_end': generated_pos[:, 1],     
                'z_end': generated_pos[:, 2],     
                'deg1': generated_deg[:, 0],     
                'deg2': generated_deg[:, 1],       
                'deg3': generated_deg[:, 2],       
                'deg4': generated_deg[:, 3],       
                'timestamps': np.linspace(0, 1, len(generated_pos)) 
            })

            results = {
                'trajectory_type': trajectory_type,
                'num_points': len(generated_df),
                'transformer_used': True,
                'applied_weight': weight if weight is not None else 'adaptive'
            }

            print("생성된 궤적에 스무딩 적용 중...")
            generated_df = self.smooth_data(generated_df, smoothing_factor=0.5)

            print(f"궤적 생성 완료: {len(generated_df)} 포인트")
            return generated_df, results

        except Exception as e:
            print(f"궤적 생성 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def visualize_trajectories(self, target_df, user_df, generated_df, trajectory_type, show=True):
        """궤적 시각화 (타겟, 사용자, 생성된 궤적)"""
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2]) 
        
        # 3D 그래프 생성
        ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
        
        # 타겟 궤적의 엔드이펙터 좌표 
        target_degrees = target_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        target_endpoints = np.array([calculate_end_effector_position(deg) for deg in target_degrees]) * 1000
        target_x, target_y, target_z = target_endpoints[:, 0], target_endpoints[:, 1], target_endpoints[:, 2]
        
        print("target_df.columns:", target_df.columns)
        print("target_df.head():\n", target_df.head())

        # 사용자 궤적의 엔드이펙터 좌표
        user_degrees = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        user_endpoints = np.array([calculate_end_effector_position(deg) for deg in user_degrees]) * 1000
        user_x, user_y, user_z = user_endpoints[:, 0], user_endpoints[:, 1], user_endpoints[:, 2]
        
        # 생성된 궤적의 엔드이펙터 좌표
        gen_x = generated_df['x_end'].values 
        gen_y = generated_df['y_end'].values 
        gen_z = generated_df['z_end'].values 
        
        # 궤적 그리기
        ax_3d.plot(target_x, target_y, target_z, 'b-', linewidth=2, label='Target Trajectory')
        ax_3d.plot(user_x, user_y, user_z, 'r-', linewidth=2, label='User Trajectory')
        ax_3d.plot(gen_x, gen_y, gen_z, 'g-', linewidth=2, label='Generated Trajectory')
        
        # 3D 그래프 설정
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title('End-effector Trajectory')
        ax_3d.legend()
        
        # 오른쪽 영역을 2x2 서브플롯으로 분할
        gs_right = gs[0, 1].subgridspec(2, 2)
        
        # 시간축 생성
        t_target = np.linspace(0, 200, len(target_df))  
        t_user = np.linspace(0, 200, len(user_df))
        t_gen = np.linspace(0, 200, len(generated_df))
        
        # 관절 각도 그래프 (4개의 관절 각도를 2x2 서브플롯으로)
        joint_names = ['deg1', 'deg2', 'deg3', 'deg4']
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  # 2x2 그리드 위치
        
        for i, (joint_name, pos) in enumerate(zip(joint_names, positions)):
            # 각 관절을 위한 서브플롯 생성
            ax = fig.add_subplot(gs_right[pos])
            
            # 각 관절 데이터 플로팅
            if joint_name in target_df.columns:
                ax.plot(t_target, target_df[joint_name].values, 'b-', linewidth=1.5, label='Target')
            
            if joint_name in user_df.columns:
                ax.plot(t_user, user_df[joint_name].values, 'r-', linewidth=1.5, label='User')
            
            if joint_name in generated_df.columns:
                ax.plot(t_gen, generated_df[joint_name].values, 'g-', linewidth=1.5, label='Generated')
            
            # 그래프 설정
            ax.set_xlabel('timestamp')
            ax.set_ylabel('deg')
            ax.set_title(joint_name)
            ax.grid(True)
            ax.legend()
        
        # 전체 제목 추가
        plt.suptitle(f'Trajectory type: {trajectory_type}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  

        # 그래프 표시
        if show:
            plt.show()
        else:
            plt.close()

def main():
    print("\n======== Trajectory Generation Mode ========")

    base_dir = "data"
    model_path = "best_trajectory_transformer.pth"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return False

    try:
        analyzer = TrajectoryAnalyzer(
            classification_model="best_classification_model.pth",
            base_dir=base_dir
        )
    except Exception as e:
        print(f"Error: Failed to initialize analyzer: {str(e)}")
        return False

    generator = EndeffectorGenerator(analyzer, model_path=model_path)
    evaluator = TrajectoryEvaluator()

    non_golden_dir = os.path.join(base_dir, "non_golden_sample")
    non_golden_files = [f for f in os.listdir(non_golden_dir) if f.endswith('.txt')]

    if not non_golden_files:
        print("User trajectory file not found.")
        return False

    selected_file = random.choice(non_golden_files)
    user_path = os.path.join(non_golden_dir, selected_file)
    user_trajectory, trajectory_type = analyzer.load_user_trajectory(user_path)
    print(f"\nSelected User Trajectory : {trajectory_type}")

    golden_dir = os.path.join(base_dir, "golden_sample")
    golden_files = [f for f in os.listdir(golden_dir) if trajectory_type in f and f.endswith('.txt')]

    if not golden_files:
        print(f"{trajectory_type} could not find target trajectory of type .")
        return False

    target_file = golden_files[0]
    print(f"Matched Target Trajectory: {target_file}")
    # target_path = os.path.join(golden_dir, target_file)
    target_trajectory, target_file = analyzer.load_target_trajectory(trajectory_type)

    evaluation_result = evaluator.evaluate_trajectory(user_trajectory, trajectory_type)
    golden_dict = load_golden_evaluation_results(trajectory_type, base_dir)
    final_score = calculate_score_with_golden(evaluation_result, golden_dict)
    grade = convert_score_to_rank(final_score)
    weight = generator.grade_to_weight.get(grade, 0.8)

    print(f"[Final Score] => {final_score:.2f} / 100")
    print(f"[Final Grade] => {grade}등급")
    print(f"[Applied Interpolation Weight] => {weight:.2f}")

    print("\nGenerating Trajectories With Transformer Models...")
    generated_df, results = generator.generate_trajectory(
        target_df=target_trajectory, 
        user_df=user_trajectory, 
        trajectory_type=trajectory_type,
        weight=weight
    )

    if generated_df is None:
        print("Trajectory creation failed")
        return False

    print(f"궤적 생성 완료: {len(generated_df)} 포인트")

    print("\n궤적 시각화 중...")
    generator.visualize_trajectories(
        target_df=target_trajectory,
        user_df=user_trajectory, 
        generated_df=generated_df,
        trajectory_type=trajectory_type,
        show=True
    )

    print("\n처리 완료!")
    return True

if __name__ == "__main__":
    main()
