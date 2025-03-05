import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from dataloader import ClassificationDataset
from utils import *
import random
import os
import math

# 분류 모델 및 궤적 로드
class TrajectoryAnalyzer:
    def __init__(self, classification_model: str = "best_classification_model.pth", base_dir="data"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        c_base_path = "data/all_data"
        self.c_dataset = ClassificationDataset(c_base_path)  
        print("Available labels", self.c_dataset.unique_labels)

        self.trajectory_types = {i: label for i, label in enumerate(self.c_dataset.unique_labels)}
        # print("\nGenerated trajectory_types:", self.trajectory_types)

        self.classifier = self.load_classifier(classification_model)
        self.base_dir = base_dir
        self.golden_dir = os.path.join(self.base_dir, "golden_sample")

    def load_classifier(self, model_path : str):
        """ 분류 모델 로드 """
        try:
            from model import TransformerModel
            model = TransformerModel(
                input_dim=21,      
                d_model=32,       
                nhead=2,           
                num_layers=3,      
                num_classes=len(self.trajectory_types)
            ).to(self.device)

            # 저장된 state_dict 로드
            state_dict = torch.load(model_path, map_location=self.device)
            print("state_dict keys:", state_dict.keys())
            
            # 가중치를 모델에 적용
            model.load_state_dict(state_dict)
            
            # 평가 모드로 설정
            model.eval()
            
            print(f"Successfully loaded classification model : {model_path}")
            return model
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def load_user_trajectory(self, file_path: str = "data/non_golden_sample"):
        """ 사용자 궤적 로드 """
        try:
            df = pd.read_csv(file_path, delimiter=',')
            scaled_df, preprocessed_df = preprocess_trajectory_data(df, scaler=self.c_dataset.scaler, return_raw=True)

            # 분류 시 스케일링 적용된 데이터 사용
            tensor_data = torch.FloatTensor(scaled_df.values).unsqueeze(0)
            tensor_data = tensor_data.to(self.device)
            
            with torch.no_grad(): 
                predictions = self.classifier(tensor_data)
                predicted_class = torch.argmax(predictions, dim=1).item()
                
                # trajectory_types에서 해당 클래스 찾기
                if predicted_class in self.trajectory_types:
                    predicted_type = self.trajectory_types[predicted_class]
                else:
                    raise ValueError(f"Predicted Class Index {predicted_class}is not in the trajectory_types")
            
            print(f"Classification Result : {predicted_type}")

            return preprocessed_df, predicted_type
            
        except Exception as e:
            print(f"Trajectory file {file_path} error during processing: {str(e)}")
            raise

    def load_target_trajectory(self, trajectory_type: str):
        """ user_trajectory와 같은 타입의 target_trajectory 로드"""
        try:
            matching_files = [f for f in os.listdir(self.golden_dir) 
                            if f.startswith(trajectory_type) and f.endswith('.txt')]
            
            if not matching_files:
                raise ValueError(f"From the golden_sample directory {trajectory_type} can't find the trajectory of the type")
            
            # 매칭되는 파일들 중 하나를 무작위로 선택(타겟 궤적 하나로 수정)
            selected_file = random.choice(matching_files)
            file_path = os.path.join(self.golden_dir, selected_file)
            
            # 선택된 파일 로드 및 전처리
            df = pd.read_csv(file_path, delimiter=',')
            _, preprocessed_df = preprocess_trajectory_data(df, scaler=self.c_dataset.scaler, return_raw=True)

            
            return preprocessed_df, selected_file
            
        except Exception as e:
            print(f"Error loading target trajectory: {str(e)}")
            raise

    def validate_input(df):
        if len(df) < 3:
            raise ValueError("Insufficient data points. At least 3 points are required.")
        if not all(col in df.columns for col in ['x_end', 'y_end', 'z_end']):
            raise ValueError("Input DataFrame must contain 'x_end', 'y_end', and 'z_end' columns.")
        
    def classify_trajectory_type(self, trajectory_type: str) -> str:
        """세부 궤적 유형을 주요 궤적 유형(line, arc, circle)으로 분류"""
        if any(t in trajectory_type for t in ['d_l', 'd_r']):
            return 'line'
        elif any(t in trajectory_type for t in ['v_45', 'v_90', 'v_135', 'v_180', 'h_u', 'h_d']):
            return 'arc'
        elif any(t in trajectory_type for t in ['clock', 'counter']):
            return 'circle'
        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")

# 궤적 평가
class TrajectoryEvaluator:
    def __init__(self):
        self.trajectory_types = {
            'line': ['d_l', 'd_r'],
            'arc': {
                'vertical': ['v_45', 'v_90', 'v_135', 'v_180'],
                'horizontal': ['h_u', 'h_d']
            },
            'circle': {
                'clockwise': ['clock_big', 'clock_t', 'clock_m', 'clock_b', 'clock_l', 'clock_r'],
                'counter_clockwise': ['counter_big', 'counter_t', 'counter_m', 'counter_b', 'counter_l', 'counter_r']
            }
        }
    
    def smooth_data(self, data, R=0.02, Q=0.1):
        """칼만 필터를 사용한 데이터 스무딩"""
        n = len(data)
        smoothed = np.zeros_like(data)
        
        for dim in range(data.shape[1]):
            x_hat = data[0, dim]
            P = 1.0
            
            x_hat_full = np.array([data[0, dim], 0])
            P_full = np.eye(2)
            
            dt = 1.0
            A = np.array([[1, dt],
                         [0, 1]])
            
            H = np.array([1, 0])
            
            Q_matrix = Q * np.array([[dt**4/4, dt**3/2],
                                    [dt**3/2, dt**2]])
            
            smoothed[0, dim] = x_hat
            for k in range(1, n):
                x_hat_full = A @ x_hat_full
                P_full = A @ P_full @ A.T + Q_matrix
                
                y = data[k, dim] - H @ x_hat_full
                S = H @ P_full @ H.T + R
                K = P_full @ H.T / S
                
                x_hat_full = x_hat_full + K * y
                P_full = (np.eye(2) - np.outer(K, H)) @ P_full
                
                smoothed[k, dim] = x_hat_full[0]
        
        return smoothed
    
    ####################
    # 직선 궤적 평가
    ####################
    def evaluate_line(self, user_df):
        # 각도 데이터 추출 및 스무딩
        angle_data = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        smoothed_angles = self.smooth_data(angle_data)
        
        # End-effector 위치 계산
        user_points = np.array([calculate_end_effector_position(deg) for deg in smoothed_angles])

        def calculate_y_axis_angle(points):
            start, end = points[0], points[-1]
            dy = end[1] - start[1]
            dx = end[0] - start[0]
            dz = end[2] - start[2]
            xy_angle = np.degrees(np.arctan2(dx, dy))
            yz_angle = np.degrees(np.arctan2(dz, dy))
            return xy_angle, yz_angle
        
        def calculate_line_height(points):
            start, end = points[0], points[-1]
            height = abs(end[1] - start[1])
            return height
        
        def calculate_line_length(points):
            distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
            return np.sum(distances)
        
        line_degree = calculate_y_axis_angle(user_points)
        line_height = calculate_line_height(user_points)
        line_length = calculate_line_length(user_points)
        
        return {
            'line_degree': line_degree,
            'line_height': line_height,
            'line_length': line_length
        }
    
    ####################
    # 호 궤적 평가
    ####################
    def evaluate_arc(self, user_df):
        # 각도 데이터 추출 및 스무딩
        angle_data = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        smoothed_angles = self.smooth_data(angle_data)
        
        # End-effector 위치 계산
        user_points = np.array([calculate_end_effector_position(deg) for deg in smoothed_angles])

        def fit_circle_from_arc(points):
            p1, p2, p3 = points[0], points[len(points)//2], points[-1]
            
            # 2D 평면에서 벡터 계산
            v1 = p2[:2] - p1[:2]
            v2 = p3[:2] - p2[:2]
            
            # 중점 계산
            mid1 = (p1[:2] + p2[:2]) / 2
            mid2 = (p2[:2] + p3[:2]) / 2
            
            # 수직 벡터 계산
            n1 = np.array([-v1[1], v1[0]])
            n2 = np.array([-v2[1], v2[0]])
            
            # 정규화
            n1 = n1 / np.linalg.norm(n1)
            n2 = n2 / np.linalg.norm(n2)
            
            # 연립방정식 해결
            A = np.vstack([n1, n2])
            b = np.array([
                np.dot(n1, mid1),
                np.dot(n2, mid2)
            ])
            
            try:
                center_2d = np.linalg.solve(A, b)
                # 3D로 확장 (z 좌표는 평균값 사용)
                center = np.array([center_2d[0], center_2d[1], np.mean([p1[2], p2[2], p3[2]])])
                radius = np.linalg.norm(center - p1)
                return center, radius
            except np.linalg.LinAlgError:
                # 특이행렬인 경우 대체 방법 사용
                center = np.mean([p1, p2, p3], axis=0)
                radius = np.mean([
                    np.linalg.norm(center - p1),
                    np.linalg.norm(center - p2),
                    np.linalg.norm(center - p3)
                ])
                return center, radius

        def calculate_central_angle(points, center):
            vectors = points - center
            total_angle = 0
            for i in range(1, len(vectors)):
                v1 = vectors[i - 1]
                v2 = vectors[i]
                dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                dot = np.clip(dot, -1.0, 1.0)
                angle = np.arccos(dot)
                total_angle += angle
            return np.degrees(total_angle)

        def calculate_arc_roundness(points, center):
            distances = np.linalg.norm(points - center, axis=1)
            return np.std(distances)

        user_center, user_radius = fit_circle_from_arc(user_points)
        user_angle = calculate_central_angle(user_points, user_center)
        user_roundness = calculate_arc_roundness(user_points, user_center)

        return {
            'arc_radius': user_radius,
            'arc_angle': user_angle,
            'arc_roundness': user_roundness
        }
        
    ####################
    # 원 궤적 평가
    ####################
    def evaluate_circle(self, user_df):
        # 각도 데이터 추출 및 스무딩
        angle_data = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        smoothed_angles = self.smooth_data(angle_data)
        
        # End-effector 위치 계산
        user_points = np.array([calculate_end_effector_position(deg) for deg in smoothed_angles])

        def calculate_circle_height(points):
            return np.max(points[:, 1]) - np.min(points[:, 1])
        
        def calculate_circle_ratio(points):
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            return x_range / y_range if y_range != 0 else float('inf')
        
        def calculate_circle_radius(points):
            center = np.mean(points, axis=0)
            distances = np.linalg.norm(points - center, axis=1)
            return np.mean(distances)
        
        def calculate_start_end_distance(points):
            return np.linalg.norm(points[0] - points[-1])
        
        circle_height = calculate_circle_height(user_points)
        circle_ratio = calculate_circle_ratio(user_points)
        circle_radius = calculate_circle_radius(user_points)
        start_end_distance = calculate_start_end_distance(user_points)
        
        return {
            'circle_height': circle_height,
            'circle_ratio': circle_ratio,
            'circle_radius': circle_radius,
            'start_end_distance': start_end_distance
        }
    
    def evaluate_trajectory(self, user_trajectory, trajectory_type):
        """분류된 궤적 유형에 따른 평가 수행"""
        try:      
            # 직선 궤적 확인
            if trajectory_type in self.trajectory_types['line']:
                print("\nEvaluating line trajectory...")
                return self.evaluate_line(user_trajectory)
                
            # 호 궤적 확인
            if trajectory_type in self.trajectory_types['arc']['vertical'] or \
            trajectory_type in self.trajectory_types['arc']['horizontal']:
                print("\nEvaluating arc trajectory...")
                return self.evaluate_arc(user_trajectory)
                
            # 원 궤적 확인
            if trajectory_type in self.trajectory_types['circle']['clockwise'] or \
            trajectory_type in self.trajectory_types['circle']['counter_clockwise']:
                print("\nEvaluating circle trajectory...")
                return self.evaluate_circle(user_trajectory)
                
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")
            
        except Exception as e:
            print(f"Error during trajectory evaluation: {str(e)}")
            raise
        
############################
# (1) golden_evaluate 폴더에서
#     정답 평가 결과를 불러오는 함수
############################
def load_golden_evaluation_results(golden_type: str, base_dir: str) -> dict:
    """
    golden_evaluate 폴더에서 'golden_type.txt'를 읽어
    사전(dict) 형태로 key-value를 반환.
    예: circle_height: 0.2794
        circle_ratio: 0.5752
    """
    golden_eval_dir = os.path.join(base_dir, "golden_evaluate")
    file_name = golden_type + ".txt"
    file_path = os.path.join(golden_eval_dir, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No evaluation file found for {golden_type} at {file_path}")

    golden_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            if ':' in line:
                key, val_str = line.split(':')
                key = key.strip()
                val_str = val_str.strip()

                # 튜플인지 float인지 판별 (간단 구현)
                if val_str.startswith("(") and val_str.endswith(")"):
                    # line_degree처럼 "(-28.052, -31.2492)" 형태
                    import ast
                    try:
                        parsed_tuple = ast.literal_eval(val_str)
                        golden_dict[key] = parsed_tuple
                        continue
                    except:
                        pass  # 실패 시 아래 float 시도
                        
                try:
                    val = float(val_str)
                except ValueError:
                    val = val_str
                
                golden_dict[key] = val

    return golden_dict

############################
# (2) 사용자 vs 정답 궤적
############################
def calculate_score_with_golden(user_eval: dict,
                                golden_eval: dict,
                                use_minmax_normalize: bool = True) -> float:
    """
    1) user_eval : {'circle_height': 0.28, 'circle_ratio': 0.57, ...}
    2) golden_eval : {'circle_height': 0.2794, 'circle_ratio': 0.5752, ...}

    - (A) 공통 지표의 '차이(diff)'를 모두 구한다.
      * float 지표는 단순 절댓값 차이
      * tuple 지표(line_degree 등)는 각 원소별 절댓값 차이
    - (B) Min-Max 스케일링을 적용할지(use_minmax_normalize=True) 결정
       -> 모든 diff들을 0~1 범위로 매핑
    - (C) 노말라이즈된 diff들의 평균(avg_diff)을 계산
    - (D) 점수 = 100 - (avg_diff * 100)
      * 만약 음수가 되면 0점으로 처리
    """

    # (A) 모든 diff를 모아두는 리스트
    diff_list = []

    for key, golden_val in golden_eval.items():
        if key not in user_eval:
            continue  # 사용자 측에 없는 지표는 생략
        user_val = user_eval[key]

        # float vs float
        if isinstance(golden_val, float) and isinstance(user_val, float):
            diff = abs(golden_val - user_val)
            diff_list.append(diff)

        # tuple vs tuple (예: line_degree)
        elif isinstance(golden_val, tuple) and isinstance(user_val, tuple):
            if len(golden_val) == len(user_val):
                for i in range(len(golden_val)):
                    diff_i = abs(golden_val[i] - user_val[i])
                    diff_list.append(diff_i)

    if not diff_list:
        print("[Info] No matching metrics found.")
        return 0.0

    # (B) Min-Max 스케일링 적용 여부
    if use_minmax_normalize:
        min_diff = min(diff_list)
        max_diff = max(diff_list)
        # max_diff == min_diff 인 경우(전부 같은 값) 방어
        if abs(max_diff - min_diff) < 1e-10:
            # 차이가 전부 0이면 diff가 다 0 -> 평균도 0 -> 점수는 100
            avg_diff_normalized = 0.0
        else:
            # 모든 diff를 0~1 범위로 매핑
            normalized_diffs = [
                (d - min_diff) / (max_diff - min_diff)
                for d in diff_list
            ]
            # 평균
            avg_diff_normalized = sum(normalized_diffs) / len(normalized_diffs)
    else:
        # 노말라이즈 안 하는 경우: diff 그대로 평균
        avg_diff_normalized = sum(diff_list) / len(diff_list)

    # (C) 평균 차이를 0~100 범위로 변환
    #     예) avg_diff_normalized == 0 -> 스코어 100
    #         avg_diff_normalized == 1 -> 스코어 0
    raw_score = 100.0 - (avg_diff_normalized * 100.0)
    final_score = max(0.0, raw_score)  # 음수 방지

    print(f"[Debug] use_minmax_normalize={use_minmax_normalize}")
    print(f"[Debug] diff_list={diff_list}")
    print(f"[Debug] avg_diff_normalized={avg_diff_normalized:.4f}, raw_score={raw_score:.2f}, final_score={final_score:.2f}")

    return round(final_score, 2)

############################
# (3) 10등급으로 변환환 
############################
def convert_score_to_rank(score: float) -> int:
    """
    점수(0~100)를 10등급으로 변환:
     -  0 ~ 10  -> 10등급
     - 11 ~ 20  ->  9등급
     - ...
     - 91 ~100  ->  1등급

    반환값: 등급(1~10)
    """
    # 안전 장치
    if score < 0:
        score = 0
    elif score > 100:
        score = 100

    rank = 10 - int((score - 1) // 10)
    if rank < 1:
        rank = 1
    elif rank > 10:
        rank = 10

    return rank

# 난이도 조정기(예정)

def main():
    base_dir = os.path.join(os.getcwd(), "data")
    
    try:
        # 분석기와 평가기 초기화
        analyzer = TrajectoryAnalyzer(
            classification_model="best_classification_model.pth",
            base_dir=base_dir
        )
        evaluator = TrajectoryEvaluator()
        
        # 사용자 궤적 데이터 불러오기 및 분류
        print("\nLoading and classifying user trajectories...")
        non_golden_dir = os.path.join(base_dir, "non_golden_sample")
        non_golden_files = [f for f in os.listdir(non_golden_dir) if f.endswith('.txt')]
        
        if not non_golden_files:
            raise ValueError("No trajectory files found in the non_golden_sample directory.")
        
        # 파일 선택 및 궤적 로드
        selected_file = random.choice(non_golden_files)
        print(f"Selected user trajectory: {selected_file}")
        
        file_path = os.path.join(non_golden_dir, selected_file)
        user_trajectory, trajectory_type = analyzer.load_user_trajectory(file_path)
        
        # 궤적 평가 수행
        evaluation_result = evaluator.evaluate_trajectory(user_trajectory, trajectory_type)
        
        # 평가 결과 출력
        print("\nEvaluation Results:")
        for metric, value in evaluation_result.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            elif isinstance(value, tuple):
                rounded_values = tuple(round(v, 4) for v in value)
                print(f"{metric}: {rounded_values}")
            else:
                print(f"{metric}: {value}")
                
        # --------------------------------------------------
        # 점수 산출:
        #   1) 정답 평가 결과 불러오기
        #   2) 사용자 vs 정답 비교 -> 점수
        #   3) 등급 계산
        # --------------------------------------------------
        print("\nNow loading golden evaluation & calculating score...")
        golden_dict = load_golden_evaluation_results(trajectory_type, base_dir)
        final_score = calculate_score_with_golden(evaluation_result, golden_dict)
        
        print(f"\n[Final Score] => {final_score:.2f} / 100")

        # 등급 계산 후 출력
        grade = convert_score_to_rank(final_score)
        print(f"[Final Grade] => {grade}등급")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Please check the data directory structure and model path")

if __name__ == "__main__":
    main()