# threshold.py

import numpy as np

def compute_score(
    user_evaluation_result: dict,
    golden_evaluation_result: dict,
    trajectory_type: str
) -> float:
    """
    user_evaluation_result와 golden_evaluation_result를 직접 비교하여
    최종 단일 스코어(score)를 계산
    """
    score = 0.0

    # 1) 직선(line) 궤적인 경우
    # ---------------------------------------------------
    #   'line_degree': (xy_angle, yz_angle), 
    #   'line_height': float,
    #   'line_length': float
    # ---------------------------------------------------
    if trajectory_type in ['d_l', 'd_r']:
        user_line_deg = user_evaluation_result.get('line_degree', (0,0))
        user_line_height = user_evaluation_result.get('line_height', 0.0)
        user_line_length = user_evaluation_result.get('line_length', 0.0)

        gold_line_deg = golden_evaluation_result.get('line_degree', (0,0))
        gold_line_height = golden_evaluation_result.get('line_height', 0.0)
        gold_line_length = golden_evaluation_result.get('line_length', 0.0)

        # 각도 차이
        xy_diff = abs(user_line_deg[0] - gold_line_deg[0])
        yz_diff = abs(user_line_deg[1] - gold_line_deg[1])

        # 길이/높이 차이
        height_diff = abs(user_line_height - gold_line_height)
        length_diff = abs(user_line_length - gold_line_length)

        score = xy_diff + yz_diff + height_diff + length_diff

    # 2) 호(arc) 궤적인 경우
    # ---------------------------------------------------
    #   'arc_radius': float,
    #   'arc_angle': float,
    #   'arc_roundness': float
    # ---------------------------------------------------
    elif trajectory_type in ['v_45', 'v_90', 'v_135', 'v_180', 'h_u', 'h_d']:
        user_radius = user_evaluation_result.get('arc_radius', 0.0)
        user_angle  = user_evaluation_result.get('arc_angle', 0.0)
        user_round  = user_evaluation_result.get('arc_roundness', 0.0)

        gold_radius = golden_evaluation_result.get('arc_radius', 0.0)
        gold_angle  = golden_evaluation_result.get('arc_angle', 0.0)
        gold_round  = golden_evaluation_result.get('arc_roundness', 0.0)

        radius_diff = abs(user_radius - gold_radius)
        angle_diff  = abs(user_angle  - gold_angle)
        round_diff  = abs(user_round  - gold_round)

        score = radius_diff + angle_diff + round_diff

    # 3) 원(circle) 궤적인 경우
    # ---------------------------------------------------
    #   'circle_height': float,
    #   'circle_ratio': float,
    #   'circle_radius': float,
    #   'start_end_distance': float
    # ---------------------------------------------------
    elif trajectory_type in [
        'clock_big', 'clock_t', 'clock_m', 'clock_b', 'clock_l', 'clock_r',
        'counter_big', 'counter_t', 'counter_m', 'counter_b', 'counter_l', 'counter_r'
    ]:
        user_height = user_evaluation_result.get('circle_height', 0.0)
        user_ratio  = user_evaluation_result.get('circle_ratio', 0.0)
        user_radius = user_evaluation_result.get('circle_radius', 0.0)
        user_sedist = user_evaluation_result.get('start_end_distance', 0.0)

        gold_height = golden_evaluation_result.get('circle_height', 0.0)
        gold_ratio  = golden_evaluation_result.get('circle_ratio', 0.0)
        gold_radius = golden_evaluation_result.get('circle_radius', 0.0)
        gold_sedist = golden_evaluation_result.get('start_end_distance', 0.0)

        height_diff = abs(user_height - gold_height)
        ratio_diff  = abs(user_ratio  - gold_ratio)
        radius_diff = abs(user_radius - gold_radius)
        sedist_diff = abs(user_sedist - gold_sedist)

        score = height_diff + ratio_diff + radius_diff + sedist_diff

    else:
        raise ValueError(f"Unknown or unhandled trajectory type: {trajectory_type}")

    return score


def determine_interpolation_weight(score: float) -> float:

    # 단순 구간 매핑 예시
    if score < 10:
        return 0.3
    elif score < 20:
        return 0.5
    elif score < 30:
        return 0.7
    elif score < 40:
        return 0.9
    else:
        return 1.0

    # ------------------------------------------------------
    # 다른 방식으로 생각한게 선형 스케일링 또는 로그 스케일링
    #def determine_interpolation_weight(score: float) -> float:
    #"""
    #선형 스케일링(Linear Scaling)을 사용하여 score를 interpolation_weight로 매핑
    
    #score가 0 이하이면 min_weight,
    #score가 max_score 이상이면 max_weight로 고정하고,
    #그 중간 구간에서는 0~1 비율로 환산하여 선형 스케일
    
    #예)
    #max_score = 50.0
    #min_weight = 0.3
    #max_weight = 1.0
    
    # score가 너무 작으면 min_weight
    #if score <= 0:
    #    return min_weight
    
    # score가 너무 크면 max_weight
    #if score >= max_score:
    #    return max_weight
    
    # 0 < score < max_score 인 경우 선형 비례
    #ratio = score / max_score  # 0~1 범위
    #weight = min_weight + ratio * (max_weight - min_weight)
    #return weight
    # ------------------------------------------------------
