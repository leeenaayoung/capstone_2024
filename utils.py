import os
import csv
import random
import torch
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

scaler = StandardScaler()
##########################
# 데이터 전처리
##########################
def preprocess_trajectory_data(data_list):
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
    
    return data_v
