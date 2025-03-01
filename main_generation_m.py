import os
import argparse
import random
from analyzer import TrajectoryAnalyzer
from generation_m import ModelBasedTrajectoryGenerator
from generation_m_model import ModelTrainer

def train_model(args):
    """모델 학습 기능"""
    print("\n======== 모델 학습 모드 ========")
    
    # 학습 디렉토리 및 모델 경로 설정
    base_dir = args.base_dir
    model_path = os.path.join(args.model_dir, "joint_relationship_model.pth")
    
    # 모델 트레이너 초기화
    trainer = ModelTrainer(
        base_dir=base_dir,
        model_save_path=model_path
    )
    
    # 학습 데이터 수집 및 모델 학습
    print("\n데이터 수집 중...")
    trajectories = trainer.collect_training_data(
        n_samples=args.samples
    )
    
    if not trajectories or len(trajectories) == 0:
        print("오류: 학습 데이터를 수집할 수 없습니다. 데이터 디렉토리 구조를 확인하세요.")
        return False
    
    print(f"\n모델 학습 시작 (에포크: {args.epochs}, 배치 크기: {args.batch_size})...")
    success = trainer.train_model(
        trajectories=trajectories,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    if success:
        print(f"\n모델 학습이 성공적으로 완료되었습니다.")
        print(f"모델이 저장된 경로: {model_path}")
    else:
        print("\n모델 학습 실패")
    
    return success

def generate_trajectories(args):
    """궤적 생성 기능"""
    print("\n======== 궤적 생성 모드 ========")
    
    # 디렉토리 및 모델 경로 설정
    base_dir = args.base_dir
    model_path = os.path.join(args.model_dir, "joint_relationship_model.pth")
    
    # 모델 경로 확인
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일을 찾을 수 없습니다: {model_path}")
        print("먼저 모델을 학습시켜주세요 (--mode train)")
        return False
    
    # 분석기 초기화
    try:
        analyzer = TrajectoryAnalyzer(
            classification_model="best_classification_model.pth",
            base_dir=base_dir
        )
    except Exception as e:
        print(f"오류: 분석기 초기화 실패: {str(e)}")
        return False
    
    # 생성기 초기화
    generator = ModelBasedTrajectoryGenerator(
        analyzer=analyzer,
        model_path=model_path
    )
    
    # 사용자 궤적 파일 선택
    non_golden_dir = os.path.join(base_dir, "non_golden_sample")
    
    if not os.path.exists(non_golden_dir):
        print(f"오류: non_golden_sample 디렉토리를 찾을 수 없습니다: {non_golden_dir}")
        return False
    
    non_golden_files = [f for f in os.listdir(non_golden_dir) if f.endswith('.txt')]
    
    if not non_golden_files:
        print(f"오류: non_golden_sample 디렉토리에 궤적 파일이 없습니다: {non_golden_dir}")
        return False
    
    # 파일 선택 (지정된 파일 또는 랜덤)
    if args.trajectory_file and args.trajectory_file in non_golden_files:
        selected_file = args.trajectory_file
    else:
        selected_file = random.choice(non_golden_files)
        
    print(f"선택된 사용자 궤적 파일: {selected_file}")
    file_path = os.path.join(non_golden_dir, selected_file)
    
    # 궤적 로드 및 분류
    try:
        print("\n사용자 궤적 로드 및 분석 중...")
        user_trajectory, trajectory_type = analyzer.load_user_trajectory(file_path)
        target_trajectory, _ = analyzer.load_target_trajectory(trajectory_type)
        
        print(f"궤적 유형: {trajectory_type}")
    except Exception as e:
        print(f"오류: 궤적 로드 및 분류 실패: {str(e)}")
        return False
    
    # 관절 간 상관관계 분석
    if args.analyze_relationships:
        print("\n관절 간 상관관계 분석 중...")
        generator.analyze_joint_relationships()
    
    # 모델 기반 궤적 생성
    print("\n모델 기반 궤적 생성 중...")
    try:
        generated_df = generator.interpolate_trajectory(
            target_df=target_trajectory,
            user_df=user_trajectory,
            trajectory_type=trajectory_type
        )

        # 시각화 및 저장
        print("\n궤적 시각화 및 저장 중...")
        generator.visualize_trajectories(
            target_df=target_trajectory,
            user_df=user_trajectory,
            generated_df=generated_df,
            trajectory_type=trajectory_type,
            generation_number=args.generation_number
        )
        
        print("\n처리 완료!")
        return True
    except Exception as e:
        print(f"오류: 궤적 생성 중 오류 발생: {str(e)}")
        return False

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='궤적 생성 모델 학습 및 실행')
    
    parser.add_argument('--mode', type=str, default='generate', choices=['train', 'generate', 'both'],
                        help='실행 모드 (train: 모델 학습, generate: 궤적 생성, both: 둘 다)')
    
    # 공통 인자
    parser.add_argument('--base_dir', type=str, default='./data',
                        help='데이터 기본 디렉토리 경로')
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='모델 저장 디렉토리 경로')
    
    # 학습 관련 인자
    parser.add_argument('--epochs', type=int, default=100, 
                        help='학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='학습 배치 크기')
    parser.add_argument('--samples', type=int, default=100,
                        help='학습에 사용할 궤적 샘플 수')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='학습률')
    
    # 생성 관련 인자
    parser.add_argument('--trajectory_file', type=str, default=None,
                        help='사용할 특정 궤적 파일 이름 (지정되지 않으면 랜덤 선택)')
    parser.add_argument('--generation_number', type=int, default=1,
                        help='생성 번호 (파일 이름에 사용)')
    parser.add_argument('--analyze_relationships', action='store_true',
                        help='관절 간 상관관계 분석 수행 여부')
    
    args = parser.parse_args()
    
    # 경로 확인 및 생성
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 모드에 따른 실행
    if args.mode == 'train' or args.mode == 'both':
        train_result = train_model(args)
        if not train_result and args.mode == 'both':
            print("모델 학습 실패로 인해 궤적 생성을 건너뜁니다.")
            return
    
    if args.mode == 'generate' or args.mode == 'both':
        generate_trajectories(args)

if __name__ == "__main__":
    main()