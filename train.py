from models import *
from dataloader import *
import os
import csv
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from functools import partial
from models import TransformerModel
from models import LSTMModel
from utils import TrajectoryAnalyzer, TrajectoryGenerator
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##########################
# 분류 모델 학습
##########################
def train_classification(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # ---- Training ----
        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ---- Validation ----
        model.eval()
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_accuracy = val_correct / total_val

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_classification_model.pth')
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}")

    print("Training complete.")

##########################
# 생성 모델 학습
##########################
def train_generation(analyzer, movement_type, train_loader, val_loader, num_samples=10, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    golden_samples = []
    for _ in range(num_samples):
        try:
            target_traj, _ = analyzer.get_random_trajectory_by_type(
                analyzer.golden_dir, movement_type
            )
            golden_samples.append(target_traj)
        except Exception as e:
            print(f"Error collecting sample: {str(e)}")
    
    if not golden_samples:
        raise ValueError("No golden samples could be collected")
    
    # 데이터셋 준비
    combined_df = pd.concat(golden_samples, ignore_index=True)
    dataset = GenerationDataset(combined_df)
    
    # 모델 초기화
    input_size = len(dataset.features)
    model = LSTMModel(input_size=input_size).to(device)
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # 학습 기록
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
 
    for epoch in range(epochs):
        # 훈련
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)        
        
        # 검증
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        val_loss /= len(val_loader)
        
        # 손실 기록
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 베스트 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 모델 저장 전에 CPU로 이동
            torch.save(model.cpu().state_dict(), 'best_generation_model.pth')
            model.to(device)  # 다시 GPU로 이동
    
    # 베스트 모델 로드
    state_dict = torch.load('best_generation_model.pth')
    model.cpu()
    model.load_state_dict(state_dict)
    model.to(device)
    return model, history, dataset.scaler
    
##########################
# 두 개의 모델 학습
##########################
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # cuda 최적화
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ##########################
    # 1. 분류

    # 데이터셋 생성
    c_base_path = "data/all_data"  # TODO: 실제 폴더 경로로 수정
    c_dataset = ClassificationDataset(c_base_path)

    # 2) 고유 라벨 목록 추출
    global unique_labels
    # unique_labels = sorted(list(set(c_dataset.labels)))
    unique_labels = get_unique_labels(c_base_path)
    print("Unique labels:", unique_labels)

    # 3) 하이퍼파라미터 설정
    input_dim = 21  # 전처리 후 실제 피처 수
    d_model = 32
    nhead = 2
    num_layers = 1
    num_classes = len(unique_labels)
    num_epochs = 100
    batch_size = 32

    # 데이터셋 분할
    train_size = int(0.8 * len(c_dataset))
    val_size = int(0.1 * len(c_dataset))
    test_size = len(c_dataset) - train_size - val_size

    # Classification DataLoader
    classification_dataset = ClassificationDataset(c_base_path)
    classification_train_dataset, classification_val_dataset, classification_test_dataset = random_split(classification_dataset, [train_size, val_size, test_size])
    # classification_train_loader = DataLoader(classification_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # classification_val_loader = DataLoader(classification_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # classification_test_loader = DataLoader(classification_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    collate_fn_with_base_path = partial(collate_fn, base_path=c_base_path)

    # DataLoader에서 collate_fn을 partial로 전달
    classification_train_loader = DataLoader(classification_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_with_base_path)
    classification_val_loader = DataLoader(classification_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_with_base_path)
    classification_test_loader = DataLoader(classification_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_with_base_path)

    # 분류 모델/손실함수/옵티마이저
    model = TransformerModel(input_dim, d_model, nhead, num_layers, num_classes).to(device)
    model.apply(initialize_weights)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습/검증
    train_classification(model, classification_train_loader, classification_val_loader, criterion, optimizer, num_epochs)

    # 8) 베스트 모델 로드 후 테스트
    best_model = TransformerModel(input_dim, d_model, nhead, num_layers, num_classes).to(device)
    best_model.load_state_dict(
    torch.load('best_classification_model.pth', map_location=torch.device('cpu'))
    )
    # test_classification(best_model, classification_test_loader)

    # 단일 궤적 예측 (non_golden_sample 폴더에서 랜덤 파일 선택)
    non_golden_dir = "data/non_golden_sample"  
    txt_files = [f for f in os.listdir(non_golden_dir) if f.endswith('.txt')]

    if not txt_files:
        print(f"No .txt files found in {non_golden_dir}. Cannot do random prediction.")
    else:
        # 랜덤으로 하나 선택
        selected_file = random.choice(txt_files)
        test_file_path = os.path.join(non_golden_dir, selected_file)

        best_model.eval()
        single_data = load_and_preprocess_trajectory(test_file_path).unsqueeze(0).to(device)
        with torch.no_grad():
            out = best_model(single_data)
            _, pred_idx = torch.max(out, 1)
        # pred_label = get_unique_labels[pred_idx.item()]
        pred_label = unique_labels[pred_idx.item()]

        print(f"\n=== Random Non-Golden Prediction ===")
        print(f"Selected file: {selected_file}")
        print(f"Predicted label: {pred_label}")
    ##########################

    ##########################
    # 2. 생성

    g_data_path = "data"
    analyzer = TrajectoryAnalyzer(g_data_path)
    generator = TrajectoryGenerator()

    try:
        movement_types = analyzer.get_available_movement_types()
        print("\nTrajectory List", movement_types)
        
        if movement_types:
            # 랜덤하게 동작 타입 선택
            movement_type = random.choice(movement_types)
            
            # golden sample에서 데이터 가져오기
            target_traj, _ = analyzer.get_random_trajectory_by_type(
                analyzer.golden_dir, movement_type
            )
            
            # dataset 생성
            dataset = GenerationDataset(target_traj)
            
            # 데이터로더 생성
            g_train_loader, g_val_loader, g_test_loader = generation_dataloaders(
                target_traj,  # DataFrame 전달
                batch_size=32
            )
            
            # LSTM 모델 학습
            print("\nTraining LSTM Start...")
            model, history, scaler = train_generation(
                analyzer=analyzer,
                movement_type=movement_type,
                train_loader=g_train_loader,
                val_loader=g_val_loader,  
                num_samples=5,
                epochs=100
            )
            
            # 테스트용 궤적 로드
            print("\nLoad Test Trajectory...")
            target_traj, user_traj = analyzer.load_random_trajectories(movement_type)
            
            # LSTM으로 예측
            print("\nTrajectory Prediction...")
            test_batch = next(iter(g_test_loader))
            sequence, _ = test_batch
            sequence = sequence.to(device)
            
            model.eval()
            with torch.no_grad():
                predicted = model(sequence)
                predicted = predicted.cpu().numpy()
                predicted = scaler.inverse_transform(predicted)
                
                print("Prediction Position:", predicted[0, :3])
            
            # 시각화 비교
            print("\nTrajectory Comparison Visualization...")
            aligned_trajectory = generator.compare_trajectories(target_traj, user_traj)
            
            # 학습 결과 출력
            print("\nTraining Results:")
            print(f"Final train loss: {history['train_loss'][-1]:.4f}")
            print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
            
        else:
            print("사용 가능한 동작 타입이 없습니다.")
            
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
##########################

