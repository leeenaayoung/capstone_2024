import numpy as np
import pandas as pd
# from sympy import symbols, cos, sin, Matrix, pi

##########################
# 데이터 전처리
##########################
def preprocess_trajectory_data(data_list, scaler=None, return_raw=False):
    # DataFrame 구성
    df_t = pd.DataFrame(data_list, columns=[
        'r', 'sequence', 'timestamp', 'deg', 'deg/sec', 'mA',
        'endpoint', 'grip/rotation', 'torque', 'force', 'ori', '#'
    ])

    # 필터링: r != 's'
    df_t = df_t[df_t['r'] != 's']
    data_v = df_t.drop(['r', 'grip/rotation', '#'], axis=1)
    
    # 각 컬럼 split 처리
    splits = {
        'endpoint': ['x_end', 'y_end', 'z_end'],
        'deg': ['deg1', 'deg2', 'deg3', 'deg4'],
        'deg/sec': ['degsec1', 'degsec2', 'degsec3', 'degsec4'],
        'torque': ['torque1', 'torque2', 'torque3'],
        'force': ['force1', 'force2', 'force3'],
        'ori': ['yaw', 'pitch', 'roll']
    }
    
    for col, new_cols in splits.items():
        v_split = data_v[col].astype(str).str.split('/')
        for idx, new_col in enumerate(new_cols):
            data_v[new_col] = v_split.str.get(idx)
    
    # 원본 컬럼 제거
    data_v = data_v.drop(['deg', 'deg/sec', 'mA', 'endpoint', 'torque', 'force', 'ori'], axis=1)
    
    # 숫자 변환
    data_v = data_v.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # deg2와 deg4에서 90도 빼기
    # data_v['deg2'] = data_v['deg2'] - 90
    # data_v['deg4'] = data_v['deg4'] - 90
    
    # time 열 생성
    data_v['time'] = data_v['timestamp'] - data_v['sequence'] - 1
    data_v = data_v.drop(['sequence', 'timestamp'], axis=1)
    
    # 시간 기준 정렬
    data_v.sort_values(by=["time"], ascending=True, inplace=True)
    data_v.reset_index(drop=True, inplace=True)
    
    # 시각화 할 때 사용(스케일링 전 궤적)
    preprocessed_df = data_v.copy()

    # 데이터 스케일링
    if scaler is not None:
        scaled_data = scaler.transform(data_v.values)
        scaled_df = pd.DataFrame(scaled_data, columns=preprocessed_df.columns)
        
        if return_raw:
            return scaled_df, preprocessed_df  
        return scaled_df
    
    return data_v

def calculate_end_effector_position(degrees):
    """ degree값을 기준으로 endeffector 계산 """
    q = np.radians(degrees)
    x = (37 * np.cos(q[0]) * np.cos(q[1])) / 100 - (8 * np.sin(q[0]) * np.sin(q[3])) / 25 + \
        (8 * np.cos(q[3]) * (np.cos(q[0]) * np.cos(q[1]) * np.cos(q[2]) - \
        np.cos(q[0]) * np.sin(q[1]) * np.sin(q[2]))) / 25
    y = (37 * np.cos(q[1]) * np.sin(q[0])) / 100 + (8 * np.cos(q[0]) * np.sin(q[3])) / 25 - \
        (8 * np.cos(q[3]) * (np.sin(q[0]) * np.sin(q[1]) * np.sin(q[2]) - \
        np.cos(q[1]) * np.cos(q[2]) * np.sin(q[0]))) / 25
    z = (37 * np.sin(q[1])) / 100 + (8 * np.cos(q[3]) * (np.cos(q[1]) * np.sin(q[2]) + \
        np.cos(q[2]) * np.sin(q[1]))) / 25
    return np.array([x, y, z])

# 자코비안 계산
def calculate_jacobian_np(q1, q2, q3, q4, l1=0.26, l2=0.31):
    cos_q1, sin_q1 = np.cos(q1), np.sin(q1)
    cos_q2, sin_q2 = np.cos(q2), np.sin(q2)
    cos_q3, sin_q3 = np.cos(q3), np.sin(q3)
    cos_q4, sin_q4 = np.cos(q4), np.sin(q4)

    J = np.array([
        [l2*cos_q4*(sin_q1*sin_q2*sin_q3 - cos_q2*cos_q3*sin_q1) - l2*cos_q1*sin_q4 - l1*cos_q2*sin_q1,
         -cos_q1*(l1*sin_q2 + l2*cos_q4*(cos_q2*sin_q3 + cos_q3*sin_q2)),
         -l2*cos_q1*cos_q4*(cos_q2*sin_q3 + cos_q3*sin_q2),
         - (cos_q2*cos_q3 - sin_q2*sin_q3)*(l2*cos_q1*sin_q4 - l2*cos_q4*(sin_q1*sin_q2*sin_q3 - cos_q2*cos_q3*sin_q1)) - l2*cos_q4*(cos_q2*sin_q1*sin_q3 + cos_q3*sin_q1*sin_q2)*(cos_q2*sin_q3 + cos_q3*sin_q2)],

        [l1*cos_q1*cos_q2 - l2*sin_q1*sin_q4 + l2*cos_q4*(cos_q1*cos_q2*cos_q3 - cos_q1*sin_q2*sin_q3),
         -sin_q1*(l1*sin_q2 + l2*cos_q4*(cos_q2*sin_q3 + cos_q3*sin_q2)),
         -l2*cos_q4*sin_q1*(cos_q2*sin_q3 + cos_q3*sin_q2),
         l2*cos_q4*(cos_q1*cos_q2*sin_q3 + cos_q1*cos_q3*sin_q2)*(cos_q2*sin_q3 + cos_q3*sin_q2) - (cos_q2*cos_q3 - sin_q2*sin_q3)*(l2*sin_q1*sin_q4 - l2*cos_q4*(cos_q1*cos_q2*cos_q3 - cos_q1*sin_q2*sin_q3))],

        [0, cos_q1*(l1*cos_q1*cos_q2 - l2*sin_q1*sin_q4 + l2*cos_q4*(cos_q1*cos_q2*cos_q3 - cos_q1*sin_q2*sin_q3)) + sin_q1*(l1*cos_q2*sin_q1 + l2*cos_q1*sin_q4 - l2*cos_q4*(sin_q1*sin_q2*sin_q3 - cos_q2*cos_q3*sin_q1)),
         sin_q1*(l2*cos_q1*sin_q4 - l2*cos_q4*(sin_q1*sin_q2*sin_q3 - cos_q2*cos_q3*sin_q1)) - cos_q1*(l2*sin_q1*sin_q4 - l2*cos_q4*(cos_q1*cos_q2*cos_q3 - cos_q1*sin_q2*sin_q3)),
         - (cos_q1*cos_q2*sin_q3 + cos_q1*cos_q3*sin_q2)*(l2*cos_q1*sin_q4 - l2*cos_q4*(sin_q1*sin_q2*sin_q3 - cos_q2*cos_q3*sin_q1))
         - (cos_q2*sin_q1*sin_q3 + cos_q3*sin_q1*sin_q2)*(l2*sin_q1*sin_q4 - l2*cos_q4*(cos_q1*cos_q2*cos_q3 - cos_q1*sin_q2*sin_q3))],

        [0, sin_q1, sin_q1, -cos_q1*cos_q2*sin_q3 - cos_q1*cos_q3*sin_q2],
        [0, -cos_q1, -cos_q1, -cos_q2*sin_q1*sin_q3 - cos_q3*sin_q1*sin_q2],
        [1, 0, 0, cos_q2*cos_q3 - sin_q2*sin_q3]])

    return J

# # 자코비안 역행렬
# def calculate_jacobian_inverse(deg1, deg2, deg3, deg4, l1=0.26, l2=0.31, k_value=0.01):

#     # 라디안 변환
#     q1, q2, q3, q4 = np.radians([deg1, deg2, deg3, deg4])

#     # 자코비안 계산
#     J_np = calculate_jacobian_np(q1, q2, q3, q4)
    
#     # 자코비안 전치
#     JJT_np = np.dot(J_np, J_np.T)

#     # k2*I 계산
#     reg_term = k_value**2 * np.eye(JJT_np.shape[0])

#     # 덧셈 
#     A_np = JJT_np + reg_term

#     # A_np 역행렬 계산
#     A_inv_np = np.linalg.inv(A_np)
    
#     # 자코비안 전치에 A_np 역행렬 곱
#     J_inv_np = np.dot(J_np.T, A_inv_np)
    
#     return J_inv_np

# # Denavit-Hartenberg 변환 행렬 함수 정의
# def DH_transform(a, α, d, θ):
#     T = sp.Matrix([
#         [sp.cos(θ), -sp.sin(θ)*sp.cos(α),  sp.sin(θ)*sp.sin(α), a*sp.cos(θ)],
#         [sp.sin(θ),  sp.cos(θ)*sp.cos(α), -sp.cos(θ)*sp.sin(α), a*sp.sin(θ)],
#         [0,           sp.sin(α),            sp.cos(α),           d],
#         [0,           0,                    0,                    1]
#     ])
#     return T

# # 변환 행렬을 계산하는 함수 정의
# def calculate_dh_matrix(row):
#     l1, l2 = 0.26, 0.31
#     α1, α2, α3, α4 = sp.pi/2, 0, -sp.pi/2, 0
#     d1, d2, d3, d4 = 0, 0, 0, 0

#     # 입력된 각도를 라디안으로 변환
#     θ1_rad = sp.rad(row['deg1'])
#     θ2_rad = sp.rad(row['deg2'])
#     θ3_rad = sp.rad(row['deg3'])
#     θ4_rad = sp.rad(row['deg4'])

#     # DH 변환 행렬 계산
#     T1 = DH_transform(0, α1, d1, θ1_rad)
#     T2 = DH_transform(l1, α2, d2, θ2_rad)
#     T3 = DH_transform(0, α3, d3, θ3_rad)
#     T4 = DH_transform(l2, α4, d4, θ4_rad)

#     # 변환 행렬 계산
#     T_total = T1 * T2 * T3 * T4
#     return T_total.evalf()

# # 오일러 각 계산
# def euler_angles_from_rotation_matrix_sympy_general(R):
#     theta = sp.atan2(sp.sqrt(1 - R[2, 2]**2), R[2, 2])
#     if theta >= 0:
#         phi = sp.atan2(R[1, 2], R[0, 2])
#         psi = sp.atan2(R[2, 1], -R[2, 0])
#     else:
#         phi = sp.atan2(-R[1, 2], -R[0, 2])
#         psi = sp.atan2(-R[2, 1], R[2, 0])
#     return phi, theta, psi

# # 변환 행렬로부터 위치와 오일러 각 추출
# def extract_position_and_euler_angles(row):
#     T_total = calculate_dh_matrix(row)
#     P = T_total[:3, 3]
#     R = T_total[:3, :3]
#     # 위치 추출
#     X_position = P[0]
#     Y_position = P[1]
#     Z_position = P[2]
#     # 오일러 각 추출 (라디안 단위 유지)
#     euler_angles = euler_angles_from_rotation_matrix_sympy_general(R)
#     # 6 x 1 배열 생성
#     result = np.array([
#         [X_position],
#         [Y_position],
#         [Z_position],
#         [euler_angles[0]],
#         [euler_angles[1]],
#         [euler_angles[2]]
#     ])
#     return result.flatten()