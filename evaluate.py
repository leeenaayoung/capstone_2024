import numpy as np
import pandas as pd
from analyzer import TrajectoryAnalyzer
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from utils import *
import random
import os
import ast

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
    
    # 직선 궤적 평가
    def evaluate_line(self, user_df):
        # 각도 데이터 추출 및 스무딩
        angle_data = user_df[['deg1', 'deg2', 'deg3', 'deg4']].values
        
        # End-effector 위치 계산
        user_points = np.array([calculate_end_effector_position(deg) for deg in angle_data])
        
        # 왕복 운동에서 '정점' 구하기
        start_point = user_points[0]
        distances_from_start = np.linalg.norm(user_points - start_point, axis=1)  # 각 점까지의 거리
        turn_idx = np.argmax(distances_from_start)  # 가장 멀리 떨어진 인덱스(왕복 최정점)
        turn_point = user_points[turn_idx]

        # n각도 계산: 시작점 ~ 정점 사이 벡터 기준, XY평면과 이루는 각도
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

        # 높이 계산: (시작점 ~ 정점) z좌표 차
        def calculate_line_height(start_p, end_p):
            return abs(end_p[2] - start_p[2])

        # 궤적 전체 길이 계산
        def calculate_line_length(points):
            segment_distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
            return segment_distances.sum()

        # 결과 계산
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

    # 호 궤적 평가
    def debug_plot_3d(self, points_3d, plane_origin, ex, ey, normal, center_3d, radius_3d):
        other = getattr(self, 'other_points_3d', None)
        vecs = (other if other is not None else points_3d) - plane_origin

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection='3d')

        # 전체 궤적 점
        if other is not None:
            ax.scatter(other[:,0], other[:,1], other[:,2],
                    color='blue', s=10, alpha=0.6, label='Other Points')
        ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2],
                color='red', s=50, label='Main 3 Points')

        # 메인 평면 (ex, ey)
        plane_size = radius_3d * 1.5
        u = np.linspace(-plane_size, plane_size, 20)
        v = np.linspace(-plane_size, plane_size, 20)
        U, V = np.meshgrid(u, v)
        Pm = plane_origin + U[...,None]*ex + V[...,None]*ey
        ax.plot_surface(Pm[...,0], Pm[...,1], Pm[...,2],
                        color='cyan', alpha=0.2, label='Main Plane')

        # 수직 평면 (ey, normal)
        Pv = plane_origin + U[...,None]*ey + V[...,None]*normal
        ax.plot_surface(Pv[...,0], Pv[...,1], Pv[...,2],
                        color='magenta', alpha=0.2, label='Vertical Plane')

        # 수평 평면 (ex, normal)
        Ph = plane_origin + U[...,None]*ex + V[...,None]*normal
        ax.plot_surface(Ph[...,0], Ph[...,1], Ph[...,2],
                        color='green', alpha=0.2, label='Horizontal Plane')

        # 각 평면 위에 투영된 점들
        # Vertical projection
        s_v = vecs.dot(ey)
        t_v = vecs.dot(normal)
        proj_v = plane_origin + np.outer(s_v, ey) + np.outer(t_v, normal)
        ax.scatter(proj_v[:,0], proj_v[:,1], proj_v[:,2],
                color='magenta', marker='^', s=40, label='Proj Vert')

        # Horizontal projection
        s_h = vecs.dot(ex)
        t_h = vecs.dot(normal)
        proj_h = plane_origin + np.outer(s_h, ex) + np.outer(t_h, normal)
        ax.scatter(proj_h[:,0], proj_h[:,1], proj_h[:,2],
                color='green', marker='s', s=40, label='Proj Horz')

        # 피팅된 원
        theta = np.linspace(0, 2*np.pi, 100)
        circ = center_3d + np.outer(np.cos(theta)*radius_3d, ex) \
                        + np.outer(np.sin(theta)*radius_3d, ey)
        ax.plot(circ[:,0], circ[:,1], circ[:,2],
                color='orange', label='Fitted Arc')

        # 축 비율 맞추기
        all_pts = np.vstack([points_3d,
                            proj_v if other is not None else proj_v,
                            proj_h if other is not None else proj_h])
        xs, ys, zs = all_pts[:,0], all_pts[:,1], all_pts[:,2]
        m = np.max([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]) / 2
        cx, cy, cz = np.mean(xs), np.mean(ys), np.mean(zs)
        ax.set_xlim(cx-m, cx+m)
        ax.set_ylim(cy-m, cy+m)
        ax.set_zlim(cz-m, cz+m)

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend()
        plt.show()


    def define_plane_from_3pts(self, p1, p2, p3):
        """ 3점이 정의하는 평면 """
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
    
    # ① PCA 기반 선형성 계산 함수 추가
    def calc_linearity_pca(self, pts, origin, axis1, axis2):
        """
        pts: (N×3) 3D 점들
        origin: 평면 원점
        axis1, axis2: 투영 축 벡터 (예: ey, ez 또는 ex, ez)
        반환: 1=완벽 직선, 0=직선성 없음
        """
        V = pts - origin
        P = np.stack([V.dot(axis1), V.dot(axis2)], axis=1)   # (N×2)
        Pc = P - P.mean(axis=0)
        _, S, _ = np.linalg.svd(Pc, full_matrices=False)
        return 0.0 if S[0] < 1e-12 else float(1 - S[1]/S[0])

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
        # smoothed = self.smooth_data(angle_data)

        # (2) FK -> 3D
        all_points_3d = np.array([calculate_end_effector_position(deg) for deg in angle_data])
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
            
             # ② 선형성 계산 추가
            lin_v = self.calc_linearity_pca(
                all_segment_points, plane_origin,
                ey,  # “수직” 평면 기준 축
                normal 
            )
            lin_h = self.calc_linearity_pca(
                all_segment_points, plane_origin,
                ex,  # “수평” 평면 기준 축
                normal 
            )

            return {
                'arc_radius':            float(radius_3d),
                'arc_angle':             float(arc_angle),
                'arc_roundness':         arc_roundness_all,
                'arc_roundness_3pts':    arc_roundness_3pts,
                'linearity_vertical':    lin_v,
                'linearity_horizontal':  lin_h,
                'plane_origin':          plane_origin,
                'ex':                    ex,
                'ey':                    ey,
                'ez':                    normal,
                'center_3d':             center_3d,
                'points_3d':             triple_3
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
                'arc_radius':            going_result['arc_radius'],
                'arc_angle':             going_result['arc_angle'],
                'arc_roundness':         going_result['arc_roundness'],
                'linearity_vertical':    going_result['linearity_vertical'],
                'linearity_horizontal':  going_result['linearity_horizontal'],
            },
            'returning': {
                'arc_radius':            returning_result['arc_radius'],
                'arc_angle':             returning_result['arc_angle'],
                'arc_roundness':         returning_result['arc_roundness'],
                'linearity_vertical':    returning_result['linearity_vertical'],
                'linearity_horizontal':  returning_result['linearity_horizontal'],
            }
        }

    ####################
    # 원 궤적 평가
    ####################
    def evaluate_circle(self, user_df, visualize=False):
        angle_data = user_df[['deg1','deg2','deg3','deg4']].values
        points_3d = np.array([calculate_end_effector_position(deg) for deg in angle_data])
        if len(points_3d) < 2:
            return {k: 0.0 for k in ['circle_height','circle_ratio','circle_radius','start_end_distance','linearity_vertical','linearity_horizontal']}

        origin = points_3d.mean(axis=0)
        M = points_3d - origin
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        ez = Vt[-1] / (np.linalg.norm(Vt[-1]) + 1e-12)
        ex = Vt[0]  / (np.linalg.norm(Vt[0])  + 1e-12)
        ey = Vt[1]  / (np.linalg.norm(Vt[1])  + 1e-12)

        # 2D projection (circle metrics)
        def proj2d(pt): v=pt-origin; return np.array([v.dot(ex), v.dot(ey)])
        pts2d = np.array([proj2d(p) for p in points_3d])
        x_vals,y_vals = pts2d[:,0], pts2d[:,1]
        x_min,x_max = x_vals.min(), x_vals.max()
        y_min,y_max = y_vals.min(), y_vals.max()
        x_range,y_range = x_max-x_min, y_max-y_min

        circle_height = y_range
        circle_ratio  = (x_range/y_range) if y_range>1e-12 else 0.0
        circle_radius = (x_range+y_range)/4
        start_end_distance = np.linalg.norm(points_3d[0]-points_3d[-1])

        def calc_linearity_pca(axis):
            # points_3d, origin, ez는 evaluate_circle 내부에서 이미 계산된 것들
            V = points_3d - origin
            s = V.dot(axis)    # 주축 방향 성분
            t = V.dot(ez)      # 법선 방향 성분
            P2 = np.stack([s, t], axis=1)

            # centering
            P2_centered = P2 - P2.mean(axis=0)

            # SVD
            _, S, _ = np.linalg.svd(P2_centered, full_matrices=False)
            sigma1, sigma2 = S[0], S[1]

            # 안전장치
            if sigma1 < 1e-12:
                return 0.0

            return float(1 - sigma2/sigma1)

        linearity_vertical   = calc_linearity_pca(ey)
        linearity_horizontal = calc_linearity_pca(ex)

        if visualize:
            self.debug_plot_circle(points_3d, origin, ex, ey, ez, x_min, x_max, y_min, y_max)

        return {
            'circle_height': circle_height,
            'circle_ratio': circle_ratio,
            'circle_radius': circle_radius,
            'start_end_distance': start_end_distance,
            'linearity_vertical': linearity_vertical,
            'linearity_horizontal': linearity_horizontal
        }

    def debug_plot_circle(self, points_3d, origin, ex, ey, ez, x_min, x_max, y_min, y_max):
        vecs = points_3d - origin
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(projection='3d')

        # 1) 원래 3D 점
        ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], s=10, alpha=0.6, label='Original')

        # 2) 기준 평면 (ex,ey)
        plane_size = max(x_max-x_min, y_max-y_min)*1.2
        u = np.linspace(-plane_size, plane_size, 20)
        v = np.linspace(-plane_size, plane_size, 20)
        U,V = np.meshgrid(u, v)
        P_main = origin + U[...,None]*ex + V[...,None]*ey
        ax.plot_surface(P_main[...,0], P_main[...,1], P_main[...,2], alpha=0.2, color='cyan', label='Main Plane')

        # 3) 수직 평면: (ey, ez)
        P_vert = origin + U[...,None]*ey + V[...,None]*ez
        ax.plot_surface(P_vert[...,0], P_vert[...,1], P_vert[...,2], alpha=0.2, color='red',   label='Vertical Plane')

        # 4) 수평 평면: (ex, ez)
        P_horiz = origin + U[...,None]*ex + V[...,None]*ez
        ax.plot_surface(P_horiz[...,0], P_horiz[...,1], P_horiz[...,2], alpha=0.2, color='green', label='Horizontal Plane')

        # 5) 투영된 점들: Vertical plane
        s_v = vecs.dot(ey); t_v = vecs.dot(ez)
        proj_v = origin + np.outer(s_v, ey) + np.outer(t_v, ez)
        ax.scatter(proj_v[:,0], proj_v[:,1], proj_v[:,2], s=20, alpha=0.8, label='Proj Vert')

        # 6) 투영된 점들: Horizontal plane
        s_h = vecs.dot(ex); t_h = vecs.dot(ez)
        proj_h = origin + np.outer(s_h, ex) + np.outer(t_h, ez)
        ax.scatter(proj_h[:,0], proj_h[:,1], proj_h[:,2], s=20, alpha=0.8, label='Proj Horz')

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
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
# def convert_score_to_rank(score: float) -> int:
#     """
#     점수(0~100)를 10등급으로 변환:
#      -  0 ~ 10  -> 10등급
#      - 11 ~ 20  ->  9등급
#      - ...
#      - 91 ~100  ->  1등급

#     반환값: 등급(1~10)
#     """
#     # 안전 장치
#     if score < 0:
#         score = 0
#     elif score > 100:
#         score = 100

#     rank = 5 - int((score - 1) // 20)
#     if rank < 1:
#         rank = 1
#     elif rank > 10:
#         rank = 10

#     return rank
def convert_score_to_rank(score: float) -> int:
    """
    점수(0~100)를 4등급으로 변환:
     -  0 ~ 25  -> 4등급
     - 26 ~ 50  -> 3등급
     - 51 ~ 75  -> 2등급
     - 76 ~100  -> 1등급

    반환값: 등급(1~4)
    """
    # 안전 장치
    if score < 0:
        score = 0
    elif score > 100:
        score = 100

    # 점수에 따른 등급 계산
    if score <= 25:
        rank = 4
    elif score <= 50:
        rank = 3
    elif score <= 75:
        rank = 2
    else:  # score <= 100
        rank = 1

    return rank

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