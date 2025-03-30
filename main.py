import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from generation import ModelBasedTrajectoryGenerator
from evaluate import convert_score_to_rank, TrajectoryEvaluator

# 등급 기반 궤적 
class RankBasedGenerator:
    def __init__(self, analyzer, model_path=None):
        self.analyzer = analyzer
        self.model_path = model_path
        self.generator = ModelBasedTrajectoryGenerator(analyzer, model_path)
        self.evaluator = TrajectoryEvaluator()
        
    def get_weight_from_rank(self, rank):
        """ 등급에 따른 보간 가중치 계산"""
        # 가중치 맵핑
        weight_map = {
            1: 0.2,  # 1등급 - 타겟에 가깝게
            2: 0.4,  # 2등급
            3: 0.6,  # 3등급
            4: 0.8   # 4등급 - 사용자에 가깝게
        }
        
        # 범위 체크 및 기본값 설정
        if rank < 1:
            rank = 1
        elif rank > 4:
            rank = 4
            
        return weight_map[rank]
    
    def generate_trajectory_by_rank(self, target_df, user_df, trajectory_type, score=None, rank=None):
        """ 점수 또는 등급에 따라 궤적 생성 """
        # 등급 결정
        # if rank is None and score is not None:
        rank = convert_score_to_rank(score)
        # elif rank is None and score is None:
        #     # 기본값: 중간 등급
        #     rank = 2
            
        # 등급에 따른 가중치 결정
        weight = self.get_weight_from_rank(rank)
        
        print(f"생성 등급: {rank}등급 (가중치: {weight})")
        
        # 단일 가중치 배열로 설정
        weights = [weight]
        
        # 궤적 생성
        generated_df, results = self.generator.interpolate_trajectory(
            target_df, user_df, trajectory_type, weights=weights
        )
        
        # 결과 시각화
        self.generator.visualize_trajectories(
            target_df, user_df, generated_df, trajectory_type
        )
        
        return generated_df, weight
    
    def evaluate_and_generate(self, target_df, user_df, trajectory_type):
        """ 사용자 궤적 평가 후 등급에 따라 궤적 생성 """
        # 사용자 궤적 평가 수행
        print(f"\n사용자 궤적 평가중 ({trajectory_type})...")
        evaluation_result = self.evaluator.evaluate_trajectory(user_df, trajectory_type)
        
        # 정답 데이터 로드 및 점수 계산
        base_dir = os.getcwd()
        try:
            from evaluate import load_golden_evaluation_results, calculate_score_with_golden
            golden_dict = load_golden_evaluation_results(trajectory_type, base_dir)
            score = calculate_score_with_golden(evaluation_result, golden_dict)
        except Exception as e:
            print(f"정답 평가 오류: {str(e)}. 기본 점수 50으로 설정합니다.")
            score = 50.0
        
        # 등급 계산
        rank = convert_score_to_rank(score)
        print(f"\n평가 결과: {score:.2f}점 ({rank}등급)")
        
        # 등급 기반 궤적 생성
        generated_df, weight = self.generate_trajectory_by_rank(
            target_df, user_df, trajectory_type, score=score
        )
        
        return generated_df, score, rank, weight

def main():
    """평가 및 생성"""
    from analyzer import TrajectoryAnalyzer
    import random
    
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    
    # 분석기 초기화
    analyzer = TrajectoryAnalyzer(
        classification_model="best_classification_model.pth",
        base_dir=base_dir
    )
    
    # 등급 기반 생성기 초기화
    rank_generator = RankBasedGenerator(
        analyzer,
        model_path="best_generation_model.pth"
    )
    
    # 사용자 궤적 파일 랜덤 선택
    non_golden_dir = os.path.join(data_dir, "non_golden_sample")
    non_golden_files = [f for f in os.listdir(non_golden_dir) if f.endswith('.txt')]
    
    if not non_golden_files:
        print("사용자 궤적 파일을 찾을 수 없습니다.")
        return
    
    # 랜덤하게 하나의 사용자 궤적 선택
    selected_file = random.choice(non_golden_files)
    print(f"선택된 사용자 궤적: {selected_file}")
    
    # 사용자 궤적 로드 및 분류
    user_path = os.path.join(non_golden_dir, selected_file)
    user_df, trajectory_type = analyzer.load_user_trajectory(user_path)
    
    # 해당 타입의 타겟 궤적 찾기
    golden_dir = os.path.join(data_dir, "golden_sample")
    golden_files = [f for f in os.listdir(golden_dir) if trajectory_type in f and f.endswith('.txt')]
    
    if not golden_files:
        print(f"{trajectory_type} 타입의 타겟 궤적을 찾을 수 없습니다.")
        return
    
    # 타겟 궤적 로드
    target_file = golden_files[0]
    print(f"매칭된 타겟 궤적: {target_file}")
    target_path = os.path.join(golden_dir, target_file)
    target_df, _ = analyzer.load_user_trajectory(target_path)
    
    # 등급에 따른 궤적 생성 (자동 평가)
    print("\n사용자 궤적 평가 및 등급 기반 생성 시작...")
    generated_df, score, rank, weight = rank_generator.evaluate_and_generate(
        target_df, user_df, trajectory_type
    )
    
    print(f"\n평가 결과: {score:.2f}점 ({rank}등급)")
    print(f"적용된 보간 가중치: {weight}")
    print("\n궤적 생성 완료!")

if __name__ == "__main__":
    main()