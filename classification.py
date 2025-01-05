import os
import csv
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

##########################
# 1. 기본 설정 및 시드 고정
##########################
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

##########################
# 2. 전역 스케일러 (예시)
##########################
scaler = StandardScaler()

##########################
# 3. TrajectoryDataset
##########################
class TrajectoryDataset(Dataset):
    def __init__(self, base_path):
        self.data = []
        self.labels = []
        self.load_data(base_path)

        # 모든 데이터를 모은 후, 스케일링 적용
        all_data = torch.cat(self.data, dim=0)  # (row-wise) 통합
        scaler.fit(all_data)  # 스케일러에 fit

        # 스케일링 후 다시 저장
        scaled_data_list = []
        for sample in self.data:
            # sample: shape (seq_len, feature_dim)
            # scaler.transform 은 (N, D) 형태가 필요하므로
            sample_np = sample.numpy()
            sample_scaled = scaler.transform(sample_np)
            scaled_data_list.append(torch.tensor(sample_scaled, dtype=torch.float32))

        self.data = scaled_data_list  # 최종 교체

    def load_data(self, base_path):
        """
        base_path 아래에 여러 '라벨 폴더'가 있고,
        각 폴더에 .txt 파일들이 있다고 가정.
        폴더 이름 = 라벨
        """
        # 예: base_path/label1/*.txt, base_path/label2/*.txt ...
        for folder in sorted(os.listdir(base_path)):  # 폴더명 알파벳순
            folder_path = os.path.join(base_path, folder)
            if not os.path.isdir(folder_path):
                continue  # 폴더가 아니면 스킵

            for file_name in sorted(os.listdir(folder_path)):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(folder_path, file_name)

                    with open(file_path, 'r') as f:
                        reader = csv.reader(f)
                        data_list = list(reader)

                    # DataFrame 구성
                    df_t = pd.DataFrame(data_list)
                    df_t.columns = [
                        'r', 'sequence', 'timestamp', 'deg', 'deg/sec', 'mA',
                        'endpoint', 'grip/rotation', 'torque', 'force', 'ori', '#'
                    ]

                    # 필터링: r != 's'
                    df_t = df_t[df_t['r'] != 's']

                    # 필요없는 컬럼 삭제
                    data_v = df_t.drop(['r', 'grip/rotation', '#'], axis=1)

                    # endpoint split
                    v_split = data_v['endpoint'].astype(str).str.split('/')
                    data_v['x_end'] = v_split.str.get(0)
                    data_v['y_end'] = v_split.str.get(1)
                    data_v['z_end'] = v_split.str.get(2)

                    # deg split
                    v_split = data_v['deg'].astype(str).str.split('/')
                    data_v['deg1'] = v_split.str.get(0)
                    data_v['deg2'] = v_split.str.get(1)
                    data_v['deg3'] = v_split.str.get(2)
                    data_v['deg4'] = v_split.str.get(3)

                    # deg/sec split
                    v_split = data_v['deg/sec'].astype(str).str.split('/')
                    data_v['degsec1'] = v_split.str.get(0)
                    data_v['degsec2'] = v_split.str.get(1)
                    data_v['degsec3'] = v_split.str.get(2)
                    data_v['degsec4'] = v_split.str.get(3)

                    # torque split
                    v_split = data_v['torque'].astype(str).str.split('/')
                    data_v['torque1'] = v_split.str.get(0)
                    data_v['torque2'] = v_split.str.get(1)
                    data_v['torque3'] = v_split.str.get(2)

                    # force split
                    v_split = data_v['force'].astype(str).str.split('/')
                    data_v['force1'] = v_split.str.get(0)
                    data_v['force2'] = v_split.str.get(1)
                    data_v['force3'] = v_split.str.get(2)

                    # ori split
                    v_split = data_v['ori'].astype(str).str.split('/')
                    data_v['yaw'] = v_split.str.get(0)
                    data_v['pitch'] = v_split.str.get(1)
                    data_v['roll'] = v_split.str.get(2)

                    # 원본 컬럼 제거
                    data_v = data_v.drop(
                        ['deg', 'deg/sec', 'mA', 'endpoint', 'torque', 'force', 'ori'],
                        axis=1
                    )

                    # 숫자 변환
                    data_v = data_v.apply(pd.to_numeric, errors='coerce').fillna(0)

                    # time 열 생성
                    data_v['time'] = data_v['timestamp'] - data_v['sequence'] - 1
                    data_v = data_v.drop(['sequence', 'timestamp'], axis=1)

                    # 시간 기준 정렬
                    data_v.sort_values(by=["time"], ascending=True, inplace=True)
                    data_v.reset_index(drop=True, inplace=True)

                    # 텐서 변환
                    tensor_data = torch.tensor(data_v.values, dtype=torch.float32)
                    self.data.append(tensor_data)
                    self.labels.append(folder)  # 폴더 이름을 라벨로

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 라벨 인덱스로 변환은 collate_fn에서 처리
        return self.data[idx], self.labels[idx]

##########################
# 4. 라벨 목록 추출
##########################
def get_unique_labels(base_path):
    """주어진 base_path 폴더 아래 폴더명을 라벨로 보고, 모든 txt를 읽어서 labels 를 모은 뒤, 정렬 반환."""
    tmp_dataset = TrajectoryDataset(base_path)
    return sorted(list(set(tmp_dataset.labels)))

##########################
# 5. collate_fn
##########################
def collate_fn(batch):
    # batch: list of (data_tensor, label_str)
    data_list = [item[0] for item in batch]
    label_list = [item[1] for item in batch]

    # pad_sequence -> (batch_size, max_len, feature_dim)
    data_padded = pad_sequence(data_list, batch_first=True, padding_value=0)

    # 라벨 인덱스 변환
    label_indices = [unique_labels.index(lbl) for lbl in label_list]
    label_indices = torch.tensor(label_indices, dtype=torch.long)

    return data_padded, label_indices

##########################
# 6. 모델 정의
##########################
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes, max_len=2000):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)

        # 위치 인코딩 (파라미터)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, d_model))

        # Transformer 인코더
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        seq_len = x.size(1)

        # 위치 인코딩
        pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)

        # 임베딩
        x = self.embedding(x) + pos_enc
        # Transformer는 (seq_len, batch, d_model) 형태를 기본으로 함
        x = x.permute(1, 0, 2)  # -> [seq_len, batch, d_model]

        x = self.transformer_encoder(x)  # [seq_len, batch, d_model]
        x = x.mean(dim=0)                # [batch, d_model]
        out = self.fc(x)                 # [batch, num_classes]
        return out

##########################
# 7. 학습/검증 루프
##########################
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
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
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}")

    print("Training complete.")

##########################
# 8. 테스트 함수
##########################
def test_model(model, test_loader):
    model.eval()
    test_correct = 0
    total_test = 0
    all_labels_list = []
    all_preds_list = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            test_correct += (predicted == labels).sum().item()
            total_test += labels.size(0)

            all_labels_list.extend(labels.cpu().numpy())
            all_preds_list.extend(predicted.cpu().numpy())

    test_acc = test_correct / total_test
    print(f"Test Accuracy: {test_acc:.4f}")

    # 혼동 행렬
    cm_normalized = confusion_matrix(all_labels_list, all_preds_list, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=unique_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix")
    plt.show()

##########################
# 9. 단일 궤적 예측 함수
##########################
def load_and_preprocess_trajectory(file_path):
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
    arr_scaled = scaler.transform(data_v.values)
    return torch.tensor(arr_scaled, dtype=torch.float32)

##########################
# 10. 메인 실행부
##########################
if __name__ == '__main__':
    # 1) 데이터셋 생성
    base_path = "data/all_data"  # TODO: 실제 폴더 경로로 수정
    dataset = TrajectoryDataset(base_path)

    # 2) 고유 라벨 목록 추출
    global unique_labels
    unique_labels = sorted(list(set(dataset.labels)))
    print("Unique labels:", unique_labels)

    # 3) 하이퍼파라미터 설정
    input_dim = 21  # 전처리 후 실제 피처 수
    d_model = 32
    nhead = 2
    num_layers = 1
    num_classes = len(unique_labels)
    num_epochs = 100
    batch_size = 32

    # 4) 데이터셋 분할
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 5) DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 6) 모델/손실함수/옵티마이저
    model = TransformerModel(input_dim, d_model, nhead, num_layers, num_classes).to(device)
    model.apply(initialize_weights)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 7) 학습/검증
    #train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs)

    # 8) 베스트 모델 로드 후 테스트
    best_model = TransformerModel(input_dim, d_model, nhead, num_layers, num_classes).to(device)
    best_model.load_state_dict(
    torch.load('best_model.pth', map_location=torch.device('cpu'))
    )
    test_model(best_model, test_loader)

    # 9) 단일 궤적 예측 (non_golden_sample 폴더에서 랜덤 파일 선택)
    non_golden_dir = "data/non_golden_sample"  # 실제 경로 맞춰주세요
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
        pred_label = unique_labels[pred_idx.item()]

        print(f"\n=== Random Non-Golden Prediction ===")
        print(f"Selected file: {selected_file}")
        print(f"Predicted label: {pred_label}")