import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import imageio.v2 as imageio
from scipy.fft import fft, fftfreq
from dtaidistance import dtw
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.ndimage import gaussian_filter1d

import os
import math
import re
import shutil
import imageio

def process_data(df, label_name):
    # 칼럼 이름 변경 (예시)
    df.columns = ['r', 'sequence', 'timestamp', 'deg', 'deg/sec', 'mA',
                  'endpoint', 'grip/rotation', 'torque', 'force', 'ori', '#']

    # 'r'이 's'인 행 제거
    df = df[df['r'] != 's']

    # 필요하지 않은 칼럼 삭제
    df = df.drop(['r', 'grip/rotation', '#'], axis=1)

    # 'endpoint' 칼럼을 x, y, z로 분리
    v_split = df['endpoint'].astype(str).str.split('/')
    df['x_end'] = v_split.str.get(0)
    df['y_end'] = v_split.str.get(1)
    df['z_end'] = v_split.str.get(2)

    # 'ori' 칼럼을 yaw, pitch, roll로 분리
    v_split = df['ori'].astype(str).str.split('/')
    df['yaw'] = v_split.str.get(0)
    df['pitch'] = v_split.str.get(1)
    df['roll'] = v_split.str.get(2)

    # deg, deg/sec, torque, force 등을 split하여 여러 컬럼으로 분할
    for col in ['deg', 'deg/sec', 'torque', 'force']:
        v_split = df[col].astype(str).str.split('/')
        parts = v_split.apply(lambda x: pd.Series(x))
        parts.columns = [f'{col}{i+1}' for i in range(parts.shape[1])]
        df = pd.concat([df, parts], axis=1)

    # 원본 칼럼 삭제
    df = df.drop(['deg', 'deg/sec', 'mA', 'endpoint', 'torque', 'force', 'ori'], axis=1)

    # 모든 칼럼을 숫자형으로 변환
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 특정 deg 컬럼 보정 (예시)
    if 'deg2' in df.columns:
        df['deg2'] = df['deg2'] - 90
    if 'deg4' in df.columns:
        df['deg4'] = df['deg4'] - 90

    # time 칼럼 생성 및 데이터 정렬
    df['time'] = df['timestamp'] - df['sequence'] - 1
    df = df.drop(['sequence', 'timestamp'], axis=1)
    df = df.sort_values(by=["time"], ascending=True).reset_index(drop=True)

    # x_end, y_end, z_end에 가우시안 필터 적용(노이즈 완화 목적)
    for col in ['x_end', 'y_end', 'z_end']:
        df[col] = gaussian_filter1d(df[col].astype(float), sigma=5)

    # 3D 시각화 (option)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(df['x_end'], df['y_end'], df['z_end'], label=f'{label_name}')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Trajectory Visualization')
    ax.legend()
    ax.view_init(20, 80)
    plt.show()

    return df

# --------------------------------------------------------------------------
# (A) 3D 직선(Line) 평가
# --------------------------------------------------------------------------
def fit_line_3d_pca(points_3d):
    """
    points_3d: shape (N, 3)의 numpy array
    반환값: (center, direction)
        center    : 직선이 지나는 한 점 (평균점)
        direction : 직선의 방향벡터 (정규화됨)
    """
    # 1) 중심(centroid) 계산
    center = np.mean(points_3d, axis=0)

    # 2) 중심을 기준으로 평행이동
    X = points_3d - center

    # 3) SVD
    # X = U * S * V^T
    # 보통 S가 내림차순으로 정렬되며, V^T[0]이 최대 분산 방향(주성분)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    direction = Vt[0]  # shape (3,)

    # 방향 벡터 정규화
    direction /= np.linalg.norm(direction)

    return center, direction

def evaluate_line_3d(user_df, target_df):
    """
    3D 직선 피팅 및 평가:
      1) user_df, target_df에서 (x_end, y_end, z_end) 뽑아서 각각 직선 피팅
      2) 방향벡터 각도, MSE, 시작/끝점 차이 등 계산
    """
    user_points = user_df[['x_end','y_end','z_end']].values
    target_points = target_df[['x_end','y_end','z_end']].values

    # 1) 직선 피팅
    c_u, d_u = fit_line_3d_pca(user_points)
    c_t, d_t = fit_line_3d_pca(target_points)

    # 2) 방향벡터 각도
    dot_val = np.dot(d_u, d_t)
    dot_val = max(min(dot_val, 1.0), -1.0)  # floating error 방지
    angle_diff = math.degrees(math.acos(dot_val))  # degree

    # 3) MSE(수직거리 제곱평균): user_points -> target 직선
    dist_list = []
    for p in user_points:
        # target 직선에 사영
        alpha = np.dot((p - c_t), d_t)
        q = c_t + alpha*d_t
        dist = np.linalg.norm(p - q)  # 수직거리
        dist_list.append(dist)
    mse_line = np.mean(np.array(dist_list)**2)

    # 4) 시작/끝점 차이 (원하면)
    start_diff = np.linalg.norm(user_points[0] - target_points[0])
    end_diff   = np.linalg.norm(user_points[-1] - target_points[-1])

    return {
        'angle_diff': angle_diff,
        'line_mse':  mse_line,
        'start_diff': start_diff,
        'end_diff':  end_diff
    }

# --------------------------------------------------------------------------
# (B) 3D 평면(Plane) 피팅 및 투영 -> 2D
# --------------------------------------------------------------------------
def fit_plane_3d(points_3d):
    """
    3D 점들에 대해 PCA로 '최적 평면'의 법선벡터, 그리고 평면이 지나는 한 점(centroid) 구함
    """
    center = np.mean(points_3d, axis=0)
    X = points_3d - center

    # SVD
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    # 최소 singular value에 해당하는 고유벡터가 평면의 법선
    normal = Vt[-1]
    normal /= np.linalg.norm(normal)

    return center, normal

def project_points_onto_plane(points_3d, plane_center, plane_normal):
    """
    3D 점들을 (plane_center, plane_normal)로 정의된 평면에 사영하여 2D 좌표로 변환
    반환: (xy_points, u_axis, v_axis)
      - xy_points: (N, 2) 모양의 numpy array
      - u_axis, v_axis: 평면 내를 이루는 직교(orthogonal) 3D 벡터
    """
    # 1) plane_normal에 수직인 임의 벡터 구하기
    z_axis = np.array([0,0,1], dtype=float)
    if abs(np.dot(plane_normal, z_axis)) > 0.99:
        z_axis = np.array([1,0,0], dtype=float)
    u_axis = np.cross(plane_normal, z_axis)
    u_axis /= np.linalg.norm(u_axis)

    # 2) v_axis = plane_normal x u_axis
    v_axis = np.cross(plane_normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)

    # 3) 점들을 (u_axis, v_axis) 기저로 표현
    xy_list = []
    for p in points_3d:
        vec = p - plane_center
        x_proj = np.dot(vec, u_axis)
        y_proj = np.dot(vec, v_axis)
        xy_list.append([x_proj, y_proj])
    xy_points = np.array(xy_list)

    return xy_points, u_axis, v_axis

# --------------------------------------------------------------------------
# (C) 2D Circle(원) & Arc(호) 피팅
# --------------------------------------------------------------------------
def fit_circle_algebraic(x, y):
    """
    2D 원 방정식 x^2 + y^2 + D*x + E*y + F = 0
    -> (center_x, center_y, R) 반환
    """
    A = np.column_stack((x, y, np.ones_like(x)))
    b = -(x**2 + y**2)

    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = sol
    cx = -D / 2.0
    cy = -E / 2.0
    R = math.sqrt(cx**2 + cy**2 - F)
    return cx, cy, R

def fit_arc_2d(x, y):
    """
    2D arc 피팅: 원 + 각도 범위
    (cx, cy, R), 각점에 대한 atan2 -> min~max로 start/end
    """
    cx, cy, R = fit_circle_algebraic(x, y)
    angles = np.degrees(np.arctan2(y - cy, x - cx))
    start_theta = angles.min()
    end_theta   = angles.max()
    return cx, cy, R, start_theta, end_theta

# --------------------------------------------------------------------------
# (D) 3D Arc 평가
# --------------------------------------------------------------------------
def fit_arc_3d(user_points_3d, target_points_3d):
    """
    3D에서 Arc(호)를 피팅하기 위해:
      1) user 3D 점들로 평면 피팅
      2) user, target 모두 사영 -> 2D (arc)
      3) (cx, cy, R, start/end) 비교
    """
    center_u, normal_u = fit_plane_3d(user_points_3d)

    # user -> 2D
    user_2d, u_ax, v_ax = project_points_onto_plane(user_points_3d, center_u, normal_u)
    # target -> 2D
    target_2d, _, _     = project_points_onto_plane(target_points_3d, center_u, normal_u)

    # arc 피팅
    cx_u, cy_u, R_u, s_u, e_u = fit_arc_2d(user_2d[:,0], user_2d[:,1])
    cx_t, cy_t, R_t, s_t, e_t = fit_arc_2d(target_2d[:,0], target_2d[:,1])

    # arc 차이
    radius_diff = abs(R_u - R_t)
    arc_u = abs(e_u - s_u)
    arc_t = abs(e_t - s_t)
    angle_diff = abs(arc_u - arc_t)

    # 3D 중심점
    C3d_u = center_u + cx_u*u_ax + cy_u*v_ax
    C3d_t = center_u + cx_t*u_ax + cy_t*v_ax
    center_diff_3d = np.linalg.norm(C3d_u - C3d_t)

    return {
        'radius_diff': radius_diff,
        'angle_diff': angle_diff,
        'center_diff_3d': center_diff_3d
    }

# --------------------------------------------------------------------------
# (E) 3D Circle(원) 평가
# --------------------------------------------------------------------------
def fit_circle_3d(user_points_3d, target_points_3d):
    """
    3D 상의 원(Circle) 평가
      1) user 3D 평면 -> 사영 -> 2D circle fit
      2) target도 동일 평면에 사영 -> 2D circle fit
      3) 반지름, 중심점 차이 등 계산
    """
    # user 평면
    center_u, normal_u = fit_plane_3d(user_points_3d)

    # 사영
    user_2d, u_ax, v_ax = project_points_onto_plane(user_points_3d, center_u, normal_u)
    target_2d, _, _     = project_points_onto_plane(target_points_3d, center_u, normal_u)

    # 2D circle fit
    x_u, y_u = user_2d[:,0], user_2d[:,1]
    cx_u, cy_u, R_u = fit_circle_algebraic(x_u, y_u)

    x_t, y_t = target_2d[:,0], target_2d[:,1]
    cx_t, cy_t, R_t = fit_circle_algebraic(x_t, y_t)

    # 반지름 차이
    radius_diff = abs(R_u - R_t)

    # 3D 중심점 복원
    center3d_u = center_u + cx_u*u_ax + cy_u*v_ax
    center3d_t = center_u + cx_t*u_ax + cy_t*v_ax
    center_diff_3d = np.linalg.norm(center3d_u - center3d_t)

    return {
        'radius_diff': radius_diff,
        'center_diff_3d': center_diff_3d
    }

# --------------------------------------------------------------------------
# (F) 3D DTW 평가 (Arc, Circle, Line 무엇이든 3D 시퀀스면 가능)
# --------------------------------------------------------------------------
def evaluate_arc_dtw_3d(user_df, target_df):
    """
    3D DTW를 단순화해, x,y,z 각각 DTW한 뒤 평균
    """
    user_points_3d = user_df[['x_end','y_end','z_end']].values
    target_points_3d = target_df[['x_end','y_end','z_end']].values

    dist_x = dtw.distance(user_points_3d[:,0], target_points_3d[:,0])
    dist_y = dtw.distance(user_points_3d[:,1], target_points_3d[:,1])
    dist_z = dtw.distance(user_points_3d[:,2], target_points_3d[:,2])

    dtw_3d = (dist_x + dist_y + dist_z) / 3.0
    return dtw_3d

# --------------------------------------------------------------------------
# (G) 실행 예시
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # (1) 원본 데이터 불러오기
    df_d_l = pd.read_csv('/content/drive/MyDrive/캡스톤/3차 데이터/golden_sample/d_l_gold/d_l_1.txt', delimiter=',')
    d_l_path = pd.read_csv('/content/drive/MyDrive/캡스톤/3차 데이터/non_golden_sample/d_l/d_l_10.txt', delimiter = ',')

    # (2) process_data() 수행
    user_df = process_data(df_d_l, label_name="User Traj")
    target_df = process_data(d_l_path, label_name="Target Traj")

    # (3) 3D 평가 함수 호출
    #     1) 직선
    line_res = evaluate_line_3d(user_df, target_df)
    print("Line 3D Evaluation:", line_res)

    #     2) Arc
    arc_res = fit_arc_3d(
        user_df[['x_end','y_end','z_end']].values,
        target_df[['x_end','y_end','z_end']].values
    )
    print("Arc 3D Evaluation:", arc_res)

    #     3) Circle
    circle_res = fit_circle_3d(
        user_df[['x_end','y_end','z_end']].values,
        target_df[['x_end','y_end','z_end']].values
    )
    print("Circle 3D Evaluation:", circle_res)

    #     4) DTW
    dtw_3d_val = evaluate_arc_dtw_3d(user_df, target_df)
    print("3D DTW Distance:", dtw_3d_val)
