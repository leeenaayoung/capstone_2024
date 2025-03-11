import os
import argparse
import random
from analyzer import TrajectoryAnalyzer
from generation_m import ModelBasedTrajectoryGenerator
from generation_m_model import GenerationModel

def train_model(args):
    """ 생성 모델 학습 """
    print("\n======== Training Mode ========")
    
    # 학습 디렉토리 및 모델 경로 설정
    base_dir = args.base_dir
    model_path = "best_generation_model.pth"
    
    # 모델 트레이너 초기화
    trainer = GenerationModel(
        base_dir=base_dir,
        model_save_path=model_path
    )
    
    # 학습 데이터 수집 및 모델 학습
    trajectories = trainer.collect_training_data(
        n_samples=args.samples
    )
    
    if not trajectories or len(trajectories) == 0:
        print("Error: Could not collect training data. Check the data directory structure.")
        return False
    
    print(f"\nEpoch: {args.epochs}")
    success = trainer.train_model(
        trajectories=trajectories,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    if success:
        print(f"\nModel training has been successfully completed.")
        print(f"Model Path: {model_path}")
    else:
        print("\nModel training failure")
    
    return success

def generate_trajectories(args):
    """궤적 생성 기능"""
    print("\n======== Trajectory Generation Mode ========")
    
    # 디렉토리 및 모델 경로 설정
    base_dir = args.base_dir
    model_path = "best_generation_model.pth"
    
    # 모델 경로 확인
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return False
    
    # 분석기 초기화
    try:
        analyzer = TrajectoryAnalyzer(
            classification_model="best_classification_model.pth",
            base_dir=base_dir
        )
    except Exception as e:
        print(f"Error: Analyzer initialization failed: {str(e)}")
        return False
    
    # 생성기 초기화
    generator = ModelBasedTrajectoryGenerator(
        analyzer=analyzer,
        model_path=model_path
    )
    
    # 사용자 궤적 파일 선택
    non_golden_dir = os.path.join(base_dir, "non_golden_sample")
    
    if not os.path.exists(non_golden_dir):
        print(f"Error: non_golden_sample directory not found: {non_golden_dir}")
        return False
    
    non_golden_files = [f for f in os.listdir(non_golden_dir) if f.endswith('.txt')]
    
    if not non_golden_files:
        print(f"Error: No trajectory file in non_golden_sample directory: {non_golden_dir}")
        return False
    
    # 파일 선택 (지정된 파일 또는 랜덤)
    if args.trajectory_file and args.trajectory_file in non_golden_files:
        selected_file = args.trajectory_file
    else:
        selected_file = random.choice(non_golden_files)
        
    print(f"Selected user trajectory files: {selected_file}")
    file_path = os.path.join(non_golden_dir, selected_file)
    
    # 궤적 로드 및 분류
    try:
        print("\nLoading and analyzing user trajectories...")
        user_trajectory, trajectory_type = analyzer.load_user_trajectory(file_path)
        target_trajectory, _ = analyzer.load_target_trajectory(trajectory_type)
    except Exception as e:
        print(f"Error: Trajectory load and classification failure: {str(e)}")
        return False
    
    # 관절 간 상관관계 분석
    # print("\nAnalyzing the correlation between joints...")
    # generator.analyze_joint_relationships()
    
    
    # 모델 기반 궤적 생성
    print("\nCreating model-based trajectories...")
    try:
        generated_df = generator.interpolate_trajectory(
            target_df=target_trajectory,
            user_df=user_trajectory,
            trajectory_type=trajectory_type
        )

        # 시각화 및 저장
        print("\nVisualizing and saving trajectories...")
        generator.visualize_trajectories(
            target_df=target_trajectory,
            user_df=user_trajectory,
            generated_df=generated_df,
            trajectory_type=trajectory_type,
            generation_number=args.generation_number
        )
        
        print("\nProcessing completed!")
        return True
    except Exception as e:
        print(f"Error: Error creating trajectory: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train and execute trajectory generation models')
    
    parser.add_argument('--mode', type=str, default='generate', choices=['train', 'generate', 'both'],
                        help='Execution mode (train: model training, generate: trajectory generation, both: both modes)')
    
    # 공통 인자
    parser.add_argument('--base_dir', type=str, default='./data',
                        help='Base directory path for data')
    
    # 학습 관련 인자
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of trajectory samples for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    
    # 생성 관련 인자
    parser.add_argument('--trajectory_file', type=str, default=None,
                        help='Specific trajectory file name to use (random selection if not specified)')
    parser.add_argument('--generation_number', type=int, default=1,
                        help='Generation number (used in output filename)')
    parser.add_argument('--analyze_relationships', action='store_true',
                        help='Perform joint relationship analysis')
    
    args = parser.parse_args()
    
    # # 경로 확인 및 생성
    # os.makedirs(args.model_dir, exist_ok=True)
    
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