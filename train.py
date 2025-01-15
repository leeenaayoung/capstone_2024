from dataloader import *
import os
import csv
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from functools import partial
from model import TransformerModel, initialize_weights
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = StandardScaler()

##########################
# 분류 모델 학습
##########################
def train_classification(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_predictions = []
        actual_values = []

        # ---- Training ----
        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 예측값 계산
            _, predicted = torch.max(outputs, 1)
            
            # CPU로 이동하고 리스트로 변환하여 저장
            train_predictions.extend(predicted.cpu().numpy())
            actual_values.extend(labels.cpu().numpy())

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

        # 예측값과 실제값 비교(5개)
        # print(f"Training predictions: {train_predictions[:5]}")
        # print(f"Actual values: {actual_values[:5]}")

    print("Training complete.")

##########################
# 테스트
##########################
# def test_classification(model, test_loader):
#     model.eval()
#     test_correct = 0
#     total_test = 0
#     all_labels_list = []
#     all_preds_list = []

#     with torch.no_grad():
#         for data, labels in test_loader:
#             data, labels = data.to(device), labels.to(device)
#             outputs = model(data)
#             _, predicted = torch.max(outputs, 1)

#             test_correct += (predicted == labels).sum().item()
#             total_test += labels.size(0)

#             all_labels_list.extend(labels.cpu().numpy())
#             all_preds_list.extend(predicted.cpu().numpy())

#     test_acc = test_correct / total_test
#     print(f"Test Accuracy: {test_acc:.4f}")

#     # 혼동 행렬
#     cm_normalized = confusion_matrix(all_labels_list, all_preds_list, normalize='true')
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=unique_labels)
#     disp.plot(cmap=plt.cm.Blues)
#     plt.title("Normalized Confusion Matrix")
#     plt.show()

######################
# 단일 궤적 예측(분류 모델 테스트)
######################
def load_and_preprocess_trajectory(file_path, imported_scaler=None):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)

    df_t = pd.DataFrame(data_list, columns=[
        'r', 'sequence', 'timestamp', 'deg', 'deg/sec', 'mA',
        'endpoint', 'grip/rotation', 'torque', 'force', 'ori', '#'
    ])

    # 필터링
    df_t = df_t[df_t['r'] != 's']
    data_v = df_t.drop(['r', 'grip/rotation', '#'], axis=1)

    # endpoint
    v_split = data_v['endpoint'].astype(str).str.split('/')
    data_v['x_end'] = v_split.str.get(0)
    data_v['y_end'] = v_split.str.get(1)
    data_v['z_end'] = v_split.str.get(2)

    # deg
    v_split = data_v['deg'].astype(str).str.split('/')
    data_v['deg1'] = v_split.str.get(0)
    data_v['deg2'] = v_split.str.get(1)
    data_v['deg3'] = v_split.str.get(2)
    data_v['deg4'] = v_split.str.get(3)

    # deg/sec
    v_split = data_v['deg/sec'].astype(str).str.split('/')
    data_v['degsec1'] = v_split.str.get(0)
    data_v['degsec2'] = v_split.str.get(1)
    data_v['degsec3'] = v_split.str.get(2)
    data_v['degsec4'] = v_split.str.get(3)

    # torque
    v_split = data_v['torque'].astype(str).str.split('/')
    data_v['torque1'] = v_split.str.get(0)
    data_v['torque2'] = v_split.str.get(1)
    data_v['torque3'] = v_split.str.get(2)

    # force
    v_split = data_v['force'].astype(str).str.split('/')
    data_v['force1'] = v_split.str.get(0)
    data_v['force2'] = v_split.str.get(1)
    data_v['force3'] = v_split.str.get(2)

    # ori
    v_split = data_v['ori'].astype(str).str.split('/')
    data_v['yaw'] = v_split.str.get(0)
    data_v['pitch'] = v_split.str.get(1)
    data_v['roll'] = v_split.str.get(2)

    # 제거
    data_v = data_v.drop(['deg', 'deg/sec', 'mA', 'endpoint', 'torque', 'force', 'ori'], axis=1)

    # 숫자 변환
    data_v = data_v.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 'time' 열
    data_v['time'] = data_v['timestamp'] - data_v['sequence'] - 1
    data_v = data_v.drop(['sequence', 'timestamp'], axis=1)

    # 시간 정렬
    data_v.sort_values(by=['time'], ascending=True, inplace=True)
    data_v.reset_index(drop=True, inplace=True)

    # 스케일러 적용
    if imported_scaler is not None:
        arr_scaled = imported_scaler.transform(data_v.values)
    else:
        arr_scaled = scaler.fit_transform(data_v.values)
    return torch.tensor(arr_scaled, dtype=torch.float32)

##########################
# 실행
##########################
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # cuda 최적화
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 데이터셋 생성
    c_base_path = "data/all_data"  
    c_dataset = ClassificationDataset(c_base_path)

    # 2) 고유 라벨 목록 추출
    print("Unique labels:", c_dataset.unique_labels)

    # 3) 하이퍼파라미터 설정
    input_dim = 21 
    d_model = 32
    nhead = 2
    num_layers = 3
    num_classes = len(c_dataset.unique_labels)
    num_epochs = 100
    batch_size = 32

    # 데이터셋 분할
    train_size = int(0.8 * len(c_dataset))
    val_size = int(0.1 * len(c_dataset))
    test_size = len(c_dataset) - train_size - val_size

    # Classification DataLoader
    collate_fn_with_labels = partial(collate_fn, dataset=c_dataset)
    classification_train_dataset, classification_val_dataset, classification_test_dataset = random_split(c_dataset, [train_size, val_size, test_size])

    # DataLoader에서 collate_fn을 partial로 전달    
    classification_train_loader = DataLoader(classification_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_with_labels)
    classification_val_loader = DataLoader(classification_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_with_labels)
    classification_test_loader = DataLoader(classification_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_with_labels)

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
    # 테스트
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
        single_data = load_and_preprocess_trajectory(test_file_path, imported_scaler=c_dataset.scaler).unsqueeze(0).to(device)
        with torch.no_grad():
            out = best_model(single_data)
            _, pred_idx = torch.max(out, 1)
        # pred_label = get_unique_labels[pred_idx.item()]
        pred_label = c_dataset.unique_labels[pred_idx.item()]

        print(f"\n=== Random Non-Golden Prediction ===")
        print(f"Selected file: {selected_file}")
        print(f"Predicted label: {pred_label}")
        print("\nCompleted Classification Model")

if __name__ == "__main__":
    main()
##########################

