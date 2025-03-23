import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from dataloader import ClassificationDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import random
import os
import ast
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
        # -----------------------------------------
        # (1) 각도 데이터 추출 및 스무딩
        # -----------------------------------------
        angle_data = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        smoothed_angles = self.smooth_data(angle_data)
        
        # End-effector 위치 계산
        user_points = np.array([calculate_end_effector_position(deg) for deg in smoothed_angles])
        
        # -----------------------------------------
        # (2) 왕복 운동에서 '정점' 구하기
        #     - 시작점에서 가장 멀리 떨어진 지점(인덱스)
        # -----------------------------------------
        start_point = user_points[0]
        distances_from_start = np.linalg.norm(user_points - start_point, axis=1)  # 각 점까지의 거리
        turn_idx = np.argmax(distances_from_start)  # 가장 멀리 떨어진 인덱스(왕복 최정점)
        turn_point = user_points[turn_idx]

        # -----------------------------------------
        # (3) 각도 계산: 시작점 ~ 정점 사이 벡터 기준, XY평면과 이루는 각도
        # -----------------------------------------
        def calculate_line_angle(start_p, end_p):
            direction_vector = end_p - start_p
            dist = np.linalg.norm(direction_vector)
            if dist == 0:
                return 0.0
            # XY 평면과 이루는 각도
            # z 성분과 xy 평면에서의 거리로 atan2 사용
            angle_rad = np.arctan2(direction_vector[2], np.linalg.norm(direction_vector[:2]))
            angle_deg = np.degrees(angle_rad)
            
            # 왕복 방향에 따라 음수가 될 수 있으므로 절댓값
            return abs(angle_deg)

        # -----------------------------------------
        # (4) 높이 계산: (시작점 ~ 정점) z좌표 차
        # -----------------------------------------
        def calculate_line_height(start_p, end_p):
            return abs(end_p[2] - start_p[2])

        # -----------------------------------------
        # (5) 궤적 전체 길이 계산
        #     - 모든 연속된 점 사이의 거리 합산
        # -----------------------------------------
        def calculate_line_length(points):
            # points: [p0, p1, p2, ... , pN]
            # p0->p1, p1->p2, ... 연속된 각 구간 거리 합
            segment_distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
            return segment_distances.sum()

        # -----------------------------------------
        # (6) 결과 계산
        # -----------------------------------------
        line_degree = calculate_line_angle(start_point, turn_point)
        line_height = calculate_line_height(start_point, turn_point)
        line_length = calculate_line_length(user_points)

        return {
            'line_degree': line_degree,
            'line_height': line_height,
            'line_length': line_length
        }
        
        
    # ==============================================
    # 2D에서 "정확히 3점을 지나는 원" 구하기 (기하 공식)
    # ==============================================
    def circle_from_3points_exact_2d(self, a, b, c):
        """
        2D 점 a=(x1,y1), b=(x2,y2), c=(x3,y3)을 '정확히'
        지나는 외접원(중심, 반지름)을 구한다.
        일직선이면 None 반환.
        """
        (x1, y1), (x2, y2), (x3, y3) = a, b, c

        # 행렬식 (x1(y2-y3)+x2(y3-y1)+x3(y1-y2)) 로 일직선 판별
        delta = x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)
        if abs(delta) < 1e-12:
            # 세 점이 일직선 -> 원 정의 불가
            return None, 0.0

        # 기하 공식(외접원)
        # Ref: Circumcircle of triangle, Wikipedia etc.
        x1sqr = x1*x1
        x2sqr = x2*x2
        x3sqr = x3*x3
        y1sqr = y1*y1
        y2sqr = y2*y2
        y3sqr = y3*y3

        c_x_num = ( (x1sqr + y1sqr)*(y2 - y3)
                + (x2sqr + y2sqr)*(y3 - y1)
                + (x3sqr + y3sqr)*(y1 - y2) )
        c_x_den = 2 * delta
        cx = c_x_num / c_x_den

        c_y_num = ( (x1sqr + y1sqr)*(x3 - x2)
                + (x2sqr + y2sqr)*(x1 - x3)
                + (x3sqr + y3sqr)*(x2 - x1) )
        c_y_den = 2 * delta
        cy = c_y_num / c_y_den

        # 반지름
        r = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)
        return np.array([cx, cy]), r

        ####################
    # 호 궤적 평가
    ####################
    def debug_plot_3d(self, points_3d, plane_origin, ex, ey, ez, center_3d, radius_3d):
        """
        3D 시각화:
        - points_3d : 주된 3개 점(시작, 중간, 끝) -> 빨간색 굵은 점
        - plane_origin, ex, ey, ez : 평면 정의
        - center_3d, radius_3d    : 3D 원(arc)

        + 추가:
        - self.other_points_3d (옵션): "나머지" 좌표점들. (N,3) shape
            debug_plot_3d 호출 시 파라미터를 늘리지 않고도,
            이 값을 읽어와 작은 파란 점으로 표시함.
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # (0) "나머지" 점들(전체 궤적 등)을 작은 파란 점으로 표시
        other_points_3d = getattr(self, 'other_points_3d', None)
        if other_points_3d is not None and len(other_points_3d) > 0:
            ax.scatter(
                other_points_3d[:, 0],
                other_points_3d[:, 1],
                other_points_3d[:, 2],
                color='blue',
                s=10,   # 작은 점
                marker='o',
                label='Other Points'
            )

        # (1) 빨간색으로 3개 주요 점 (시작, 중간, 끝)
        ax.scatter(
            points_3d[:, 0],
            points_3d[:, 1],
            points_3d[:, 2],
            color='red',
            s=100,
            marker='o',
            label='Main 3 Points'
        )

        # (2) 평면 (반투명 surface)
        plane_size = max(radius_3d * 1.5, 0.5)
        n_grid = 20
        u_vals = np.linspace(-plane_size, plane_size, n_grid)
        v_vals = np.linspace(-plane_size, plane_size, n_grid)
        plane_xyz = np.zeros((n_grid, n_grid, 3))
        for i, u in enumerate(u_vals):
            for j, v in enumerate(v_vals):
                plane_xyz[i, j] = plane_origin + u*ex + v*ey

        Xp = plane_xyz[:, :, 0]
        Yp = plane_xyz[:, :, 1]
        Zp = plane_xyz[:, :, 2]
        ax.plot_surface(Xp, Yp, Zp, alpha=0.3)

        # (3) 피팅된 원(arc) 그리기
        t_samples = np.linspace(0, 2*np.pi, 100)
        circle_pts = []
        for t in t_samples:
            xx = radius_3d * np.cos(t)
            yy = radius_3d * np.sin(t)
            c3d = center_3d + xx*ex + yy*ey
            circle_pts.append(c3d)
        circle_pts = np.array(circle_pts)
        ax.plot(
            circle_pts[:, 0],
            circle_pts[:, 1],
            circle_pts[:, 2],
            color='orange',
            label='Fitted Arc'
        )

        # (4) 축 비율 맞추기
        xs = points_3d[:, 0]
        ys = points_3d[:, 1]
        zs = points_3d[:, 2]

        # 만약 other_points_3d가 있다면, 범위 계산에 포함
        if other_points_3d is not None and len(other_points_3d) > 0:
            xs = np.concatenate([xs, other_points_3d[:, 0]])
            ys = np.concatenate([ys, other_points_3d[:, 1]])
            zs = np.concatenate([zs, other_points_3d[:, 2]])

        max_range = np.array([
            xs.max() - xs.min(),
            ys.max() - ys.min(),
            zs.max() - zs.min()
        ]).max()
        mid_x = (xs.max() + xs.min()) / 2
        mid_y = (ys.max() + ys.min()) / 2
        mid_z = (zs.max() + zs.min()) / 2

        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title("Exact Circle from 3 Points (3D Visualization)")

        ax.legend(loc='best')
        plt.show()


    def define_plane_from_3pts(self, p1, p2, p3):
        """
        3점이 정의하는 평면:
          - plane_origin: 무게중심
          - normal
          - ex, ey: 평면 기저
        """
        plane_origin = (p1 + p2 + p3) / 3.0
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-12:
            # 일직선이면 None
            return plane_origin, None, None, None

        normal /= norm_len
        # ex
        ex_candidate = v1 - np.dot(v1, normal)*normal
        ex_len = np.linalg.norm(ex_candidate)
        if ex_len < 1e-12:
            # v1이 normal과 거의 평행 -> v2 시도
            ex_candidate = v2 - np.dot(v2, normal)*normal
            ex_len = np.linalg.norm(ex_candidate)
            if ex_len < 1e-12:
                return plane_origin, None, None, None
        ex = ex_candidate / ex_len

        # ey = normal x ex
        ey = np.cross(normal, ex)
        ey_norm = np.linalg.norm(ey)
        if ey_norm < 1e-12:
            return plane_origin, None, None, None
        ey /= ey_norm

        return plane_origin, normal, ex, ey

    # -------------------------------------------------
    # (C) 3D arc(원) 계산
    # -------------------------------------------------
    def circle_from_3points_exact_3d(self, p1, p2, p3):
        """
        3개의 3D 점 p1, p2, p3가 일직선이 아닌 경우,
        그 점들을 정확히 지나는 '원'의 (center_3d, radius)을 구함.
        """
        plane_origin, normal, ex, ey = self.define_plane_from_3pts(p1, p2, p3)
        if normal is None:
            # 평면 정의 불가 -> 일직선
            return None, 0.0, plane_origin, ex, ey, normal

        # 2D로 투영
        def proj_2d(pt):
            vec = pt - plane_origin
            return np.array([np.dot(vec, ex), np.dot(vec, ey)])
        p1_2d = proj_2d(p1)
        p2_2d = proj_2d(p2)
        p3_2d = proj_2d(p3)

        # 2D에서 기하 공식을 써서 원(외접원) 구하기
        c2d, r2d = self.circle_from_3points_exact_2d(p1_2d, p2_2d, p3_2d)
        if c2d is None:
            # 세 점이 일직선
            return None, 0.0, plane_origin, ex, ey, normal

        # 3D 복원
        center_3d = plane_origin + c2d[0]*ex + c2d[1]*ey
        radius_3d = r2d
        return center_3d, radius_3d, plane_origin, ex, ey, normal

    # -------------------------------------------------
    # (D) 3점에 대한 arc 각도, roundness(3점만) 계산
    # -------------------------------------------------
    def calculate_arc_metrics_3pts(self, center_3d, triple_3d):
        """
        - arc_angle: 시작~끝 벡터의 "작은 각도"(0°~180°)
        - arc_roundness: 3점의 거리 표준편차
        (방향성은 버리고, 항상 작은 호만 계산)
        """
        if len(triple_3d) != 3:
            return 0.0, 0.0

        p1, p2, p3 = triple_3d
        s_vec = p1 - center_3d  # 시작점 - 중심
        m_vec = p2 - center_3d  # 중간점 - 중심 (여기서는 각도판별 사용X)
        e_vec = p3 - center_3d  # 끝점 - 중심

        ns = np.linalg.norm(s_vec)
        nm = np.linalg.norm(m_vec)
        ne = np.linalg.norm(e_vec)
        if ns < 1e-12 or nm < 1e-12 or ne < 1e-12:
            return 0.0, 0.0

        vs = s_vec / ns
        ve = e_vec / ne

        # 내적을 이용해 시작~끝 벡터의 각도(기본 0°~180°)
        dot_val = np.clip(np.dot(vs, ve), -1.0, 1.0)
        angle_rad = np.arccos(dot_val)  # 여기선 0 ~ π 범위 (0° ~ 180°)

        # "방향성" 로직(교차곱+중간점)은 제거.
        # 대신, 혹시라도 부동소수점 오차 등으로 π(180°) 조금 넘어가는 걸 방어.
        if angle_rad > np.pi:
            # (현실적으로는 안 나오겠지만 보수적 체크)
            angle_rad = 2*np.pi - angle_rad

        # 도(deg) 변환
        arc_angle_deg = np.degrees(angle_rad)

        # 3점만의 "둥글기" (표준편차)
        dists_3 = np.linalg.norm(triple_3d - center_3d, axis=1)
        arc_roundness_3pts = float(np.std(dists_3))

        return arc_angle_deg, arc_roundness_3pts

    # -------------------------------------------------
    # (E) 왕복 궤적에서 (시작/중간/끝) 3점 뽑아 정확히 원을 구함
    # -------------------------------------------------
    def evaluate_arc(self, user_df, visualize=False):
        """
        user_df: DataFrame with columns ['deg1','deg2','deg3','deg4']
        1) 각도 -> 3D 좌표
        2) 왕복 궤적: 가장 먼 지점( turn_idx )으로 going/returning 분리
        3) 각 구간에서 (시작/중간/끝) 3점만 뽑아 정확히 원(arc) 구함
        4) arc_radius, arc_angle, 
           arc_roundness(전체 점 기준) + arc_roundness_3pts(3점만) 반환
        5) visualize=True 이면 debug_plot_3d 시각화
        """
        # (1) 스무딩
        angle_data = user_df[['deg1','deg2','deg3','deg4']].values
        smoothed = self.smooth_data(angle_data)

        # (2) FK -> 3D
        all_points_3d = np.array([calculate_end_effector_position(deg) for deg in smoothed])
        if len(all_points_3d) < 3:
            print("데이터가 3개 미만입니다.")
            return {
                'going': {'arc_radius':0.0, 'arc_angle':0.0, 'arc_roundness':0.0},
                'returning': {'arc_radius':0.0, 'arc_angle':0.0, 'arc_roundness':0.0}
            }

        # 시각화용 파란 점
        self.other_points_3d = all_points_3d

        # (3) 왕복 분리
        start_p = all_points_3d[0]
        dists = np.linalg.norm(all_points_3d - start_p, axis=1)
        turn_idx = np.argmax(dists)
        going_pts = all_points_3d[:turn_idx+1]     # 전반
        returning_pts = all_points_3d[turn_idx:]   # 후반

        # (시작/중간/끝) 3점
        def pick_3points(segment):
            if len(segment) < 3:
                if len(segment) == 2:
                    return np.array([segment[0], segment[1], segment[1]])
                else:  # 1개
                    return np.array([segment[0], segment[0], segment[0]])
            p1 = segment[0]
            p2 = segment[len(segment)//2]
            p3 = segment[-1]
            return np.array([p1, p2, p3])

        going_3 = pick_3points(going_pts)
        returning_3 = pick_3points(returning_pts)

        # (4) 구간별 3점으로 원 구하기, 전체 점 둥근 정도 계산
        def evaluate_segment(all_segment_points, triple_3):
            """
            all_segment_points: (해당 구간) 전체 점
            triple_3: 3점 (시작,중간,끝)
            """
            if len(triple_3) != 3:
                return {
                    'arc_radius': 0.0,
                    'arc_angle': 0.0,
                    'arc_roundness': 0.0,       # 전체 구간 점 기준
                    'arc_roundness_3pts': 0.0,  # 3점 기준
                    'plane_origin': np.zeros(3),
                    'ex': np.zeros(3),
                    'ey': np.zeros(3),
                    'ez': np.zeros(3),
                    'center_3d': np.zeros(3),
                    'points_3d': triple_3
                }

            # (a) 원(arc) 구하기
            center_3d, radius_3d, plane_origin, ex, ey, normal = self.circle_from_3points_exact_3d(
                triple_3[0], triple_3[1], triple_3[2]
            )
            if center_3d is None:
                return {
                    'arc_radius': 0.0,
                    'arc_angle': 0.0,
                    'arc_roundness': 0.0,
                    'arc_roundness_3pts': 0.0,
                    'plane_origin': plane_origin,
                    'ex': ex if ex is not None else np.zeros(3),
                    'ey': ey if ey is not None else np.zeros(3),
                    'ez': normal if normal is not None else np.zeros(3),
                    'center_3d': np.zeros(3),
                    'points_3d': triple_3
                }

            # (b) 3점 각도, 3점 표준편차
            arc_angle, arc_roundness_3pts = self.calculate_arc_metrics_3pts(center_3d, triple_3)

            # (c) 전체 점 편차(새로운 arc_roundness)
            dists_all = np.linalg.norm(all_segment_points - center_3d, axis=1)
            arc_roundness_all = float(np.std(dists_all))

            return {
                'arc_radius': float(radius_3d),
                'arc_angle': float(arc_angle),
                # 3점만의 편차 vs 전체 편차
                'arc_roundness_3pts': arc_roundness_3pts,     
                'arc_roundness': arc_roundness_all,           
                'plane_origin': plane_origin,
                'ex': ex,
                'ey': ey,
                'ez': normal,
                'center_3d': center_3d,
                'points_3d': triple_3
            }

        going_result = evaluate_segment(going_pts, going_3)
        returning_result = evaluate_segment(returning_pts, returning_3)

        # (5) 시각화
        if visualize:
            print("\n[DEBUG] Going arc (3 points) ...")
            self.debug_plot_3d(
                going_result['points_3d'],
                going_result['plane_origin'],
                going_result['ex'],
                going_result['ey'],
                going_result['ez'],
                going_result['center_3d'],
                going_result['arc_radius']
            )
            print("\n[DEBUG] Returning arc (3 points) ...")
            self.debug_plot_3d(
                returning_result['points_3d'],
                returning_result['plane_origin'],
                returning_result['ex'],
                returning_result['ey'],
                returning_result['ez'],
                returning_result['center_3d'],
                returning_result['arc_radius']
            )

        # (6) 결과 반환
        return {
            'going': {
                'arc_radius': going_result['arc_radius'],
                'arc_angle': going_result['arc_angle'],
                'arc_roundness': going_result['arc_roundness'],           
            },
            'returning': {
                'arc_radius': returning_result['arc_radius'],
                'arc_angle': returning_result['arc_angle'],
                'arc_roundness': returning_result['arc_roundness'],       
            }
        }

        

    ####################
    # 원 궤적 평가
    ####################
    def evaluate_circle(self, user_df, visualize=False):
        """
        1) (deg1..deg4) -> 스무딩 -> FK(3D 좌표)
        2) PCA/SVD로 평면 정의
        3) 그 평면에 모든 3D 점 투영 -> 2D bounding box 계산
        4) circle_height = y_max - y_min
        5) circle_ratio = (x_max - x_min) / (y_max - y_min)
        6) circle_radius = 여기서는 ( (x_range + y_range) / 4 ) 로 예시
           (x_range=y_range이면, circle_radius = x_range/2)
        7) start_end_distance = 3D 첫 점과 마지막 점 거리
        8) 시각화(옵션) → 투영된 평면 + 모든 점 표시
        """
        # (A) 데이터 추출 & 스무딩
        angle_data = user_df[['deg1','deg2','deg3','deg4']].values
        smoothed = self.smooth_data(angle_data)

        # (B) FK -> 3D
        points_3d = np.array([calculate_end_effector_position(deg) for deg in smoothed])
        if len(points_3d) < 2:
            print("점이 2개 미만.")
            return {
                'circle_height': 0.0,
                'circle_ratio': 0.0,
                'circle_radius': 0.0,
                'start_end_distance': 0.0
            }

        # (C) PCA/SVD로 평면 찾기
        centroid = np.mean(points_3d, axis=0)
        M = points_3d - centroid
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        plane_normal = Vt[-1, :]
        plane_normal /= (np.linalg.norm(plane_normal) + 1e-12)

        plane_origin = centroid.copy()
        ex = Vt[0, :] / (np.linalg.norm(Vt[0, :]) + 1e-12)
        ey = Vt[1, :] / (np.linalg.norm(Vt[1, :]) + 1e-12)
        ez = plane_normal

        # (D) 3D -> 2D 투영
        def project_2d(pt):
            vec = pt - plane_origin
            return np.array([np.dot(vec, ex), np.dot(vec, ey)])
        points_2d = np.array([project_2d(p) for p in points_3d])

        x_vals = points_2d[:,0]
        y_vals = points_2d[:,1]

        # (E) bounding box
        x_min, x_max = np.min(x_vals), np.max(x_vals)
        y_min, y_max = np.min(y_vals), np.max(y_vals)
        x_range = x_max - x_min
        y_range = y_max - y_min

        # (F) circle_height, circle_ratio, circle_radius
        circle_height = y_range
        circle_ratio  = (x_range / y_range) if y_range > 1e-12 else 0.0
        # 예시: 두 범위의 평균 길이/2 => ( (x_range + y_range)/2 ) / 2 = (x_range+y_range)/4
        # 다른 방식(예: (min(x_range,y_range))/2 )도 가능
        circle_radius = (x_range + y_range)/4

        # (G) start_end_distance (3D)
        start_end_distance = 0.0
        if len(points_3d) >= 2:
            start_end_distance = np.linalg.norm(points_3d[0] - points_3d[-1])

        # (H) 시각화
        if visualize:
            self.debug_plot_circle(points_3d, plane_origin, ex, ey, ez, x_min, x_max, y_min, y_max)

        return {
            'circle_height': float(circle_height),
            'circle_ratio': float(circle_ratio),
            'circle_radius': float(circle_radius),
            'start_end_distance': float(start_end_distance)
        }

    def debug_plot_circle(
        self,
        points_3d,
        plane_origin, ex, ey, ez,
        x_min, x_max, y_min, y_max
    ):
        """
        시각화(옵션):
         - points_3d (파랑 점)
         - plane_origin + ex,ey 평면 (반투명)
         - bounding box 시각화 (2D 상 사각형 -> 3D에 복원 가능)
           * 여기에 '진짜 원'은 그리지 않음 (최소제곱 피팅X),
             원이라기보다는 bounding box만 표현 가능.
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # (1) 전체 점
        ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], color='blue', s=10, label='All Points')

        # (2) 평면
        plane_size = max((x_max - x_min)*1.5, (y_max-y_min)*1.5, 0.5)
        n_grid = 20
        u_vals = np.linspace(-plane_size, plane_size, n_grid)
        v_vals = np.linspace(-plane_size, plane_size, n_grid)
        plane_xyz = np.zeros((n_grid, n_grid, 3))
        for i,u in enumerate(u_vals):
            for j,v in enumerate(v_vals):
                plane_xyz[i,j] = plane_origin + u*ex + v*ey
        Xp = plane_xyz[:,:,0]
        Yp = plane_xyz[:,:,1]
        Zp = plane_xyz[:,:,2]
        ax.plot_surface(Xp, Yp, Zp, alpha=0.3)

        # (3) bounding box를 3D로 표시(선택 사항):
        #   2D에서 x_min..x_max, y_min..y_max 사각형을 3D로 복원
        corners_2d = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
            [x_min, y_min]  # 닫기 위해
        ]
        # 3D로 변환
        corners_3d = []
        for x2d, y2d in corners_2d:
            pt3d = plane_origin + x2d*ex + y2d*ey
            corners_3d.append(pt3d)
        corners_3d = np.array(corners_3d)
        ax.plot(
            corners_3d[:,0], corners_3d[:,1], corners_3d[:,2],
            color='orange', label='Bounding Box'
        )

        # 축 비율 맞추기
        xs = points_3d[:,0]
        ys = points_3d[:,1]
        zs = points_3d[:,2]
        max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max()
        mid_x = (xs.max()+xs.min())*0.5
        mid_y = (ys.max()+ys.min())*0.5
        mid_z = (zs.max()+zs.min())*0.5
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Projection bounding box (No circle fitting)")
        ax.legend()
        plt.show()
    
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
                return self.evaluate_arc(user_trajectory, visualize=True)
                
            # 원 궤적 확인
            if trajectory_type in self.trajectory_types['circle']['clockwise'] or \
            trajectory_type in self.trajectory_types['circle']['counter_clockwise']:
                print("\nEvaluating circle trajectory...")
                return self.evaluate_circle(user_trajectory, visualize=True)
                
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")
            
        except Exception as e:
            print(f"Error during trajectory evaluation: {str(e)}")
            raise
        
##############################################
# (1) 정답 평가 결과 불러오기 (파일->dict)
##############################################
import ast

def load_golden_evaluation_results(golden_type: str, base_dir: str) -> dict:
    """
    golden_evaluate 폴더에서 golden_type+'.txt' 읽어,
    키:값 형식 -> dict로 만듦. 예:
      going: {'arc_radius': 0.6090, ...}
      returning: {'arc_radius': 0.5863, ...}
    => going, returning 둘 다 실제 dict로 파싱
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
            if ':' not in line:
                continue
            # key: val_str 분할 (첫 콜론만)
            key, val_str = line.split(':', 1)
            key = key.strip()
            val_str = val_str.strip()

            # 만약 '{...}' 형태면 dict로 파싱
            if val_str.startswith('{') and val_str.endswith('}'):
                try:
                    parsed = ast.literal_eval(val_str)
                    golden_dict[key] = parsed
                    continue
                except:
                    pass
            # float 변환 or 문자열
            try:
                val = float(val_str)
            except ValueError:
                val = val_str
            golden_dict[key] = val

    return golden_dict


##############################################
# (2) 점수 계산
##############################################
def calculate_score_with_golden(
    user_eval: dict,
    golden_eval: dict,
    use_minmax_normalize: bool = True
) -> float:
    """
    arc 예시(going/returning):
      {
        "going": {
          "arc_radius": float, "arc_angle": float, "arc_roundness": float
        },
        "returning": {
          "arc_radius": float, "arc_angle": float, "arc_roundness": float
        }
      }
    """
    diff_list = []

    # A) 순회
    for key, gold_val in golden_eval.items():
        if key not in user_eval:
            continue
        user_val = user_eval[key]

        # dict vs dict (예: "going", "returning")
        if isinstance(gold_val, dict) and isinstance(user_val, dict):
            for subkey, gv_subval in gold_val.items():
                # 여기서 subkey는 "arc_radius", "arc_angle", "arc_roundness" 등
                if subkey not in user_val:
                    continue
                uv_subval = user_val[subkey]

                # float vs float
                if isinstance(gv_subval, float) and isinstance(uv_subval, float):
                    diff_list.append(abs(gv_subval - uv_subval))

                # tuple vs tuple
                elif isinstance(gv_subval, tuple) and isinstance(uv_subval, tuple):
                    if len(gv_subval) == len(uv_subval):
                        for i in range(len(gv_subval)):
                            diff_i = abs(gv_subval[i] - uv_subval[i])
                            diff_list.append(diff_i)
            continue

        # float vs float
        if isinstance(gold_val, float) and isinstance(user_val, float):
            diff_list.append(abs(gold_val - user_val))

        # tuple vs tuple
        elif isinstance(gold_val, tuple) and isinstance(user_val, tuple):
            if len(gold_val) == len(user_val):
                for i in range(len(gold_val)):
                    diff_i = abs(gold_val[i] - user_val[i])
                    diff_list.append(diff_i)

    # B) diff_list 검증
    if not diff_list:
        print("[Info] No matching metrics found.")
        return 0.0

    # C) Min-Max 정규화 (옵션)
    if use_minmax_normalize:
        mn = min(diff_list)
        mx = max(diff_list)
        if abs(mx - mn) < 1e-12:
            avg_diff = 0.0
        else:
            ndiffs = [(d - mn)/(mx - mn) for d in diff_list]
            avg_diff = sum(ndiffs)/len(ndiffs)
    else:
        avg_diff = sum(diff_list)/len(diff_list)

    # D) 점수 환산
    raw_score = 100.0 - (avg_diff*100.0)
    final_score = max(0.0, raw_score)

    print(f"[Debug] diff_list={diff_list}")
    print(f"[Debug] avg_diff_normalized={avg_diff:.4f}, raw_score={raw_score:.2f}, final_score={final_score:.2f}")

    return round(final_score, 2)

############################
# (3) 10등급으로 변환
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

    rank = 5 - int((score - 1) // 20)
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
        #selected_file = random.choice(non_golden_files)
        selected_file = "C:\\Users\\kdh03\\Desktop\\캡스톤\\capstone_2024\\data\\all_data\\v_45_41.txt"
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