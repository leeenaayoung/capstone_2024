import numpy as np
import pandas as pd

# 궤적 데이터 전처리
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
