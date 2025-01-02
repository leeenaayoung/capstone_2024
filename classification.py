import torch
import pandas as pd 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import random
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from torch import optim
import matplotlib.pyplot as plt

# 시드 고정
def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrajectoryDataset(Dataset):
    def __init__(self, base_path):
        self.base_path = base_path
        self.golden_dir = os.path.join(base_path, "golden_sample")
        self.non_golden_dir = os.path.join(base_path, "non_golden_sample")
        self.data = []
        self.labels = []
        self.scaler = StandardScaler()
        self.model = None  

        self.classes = [
            'clock_b', 'clock_big', 'clock_l', 'clock_m', 'clock_r', 'clock_t',
            'counter_b', 'counter_big', 'counter_l', 'counter_m', 'counter_r', 
            'counter_t', 'd_l', 'd_r', 'h_d', 'h_u', 'v_135', 'v_180', 'v_45', 'v_90'
        ]
        
        # 라벨 인코더 초기화
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
        
        # 데이터 로드
        self.load_data()

    def process_data(self, df):
        """데이터 전처리"""
        # 열 이름을 수동으로 설정
        expected_columns = ['r', 'sequence', 'timestamp', 'deg', 'deg/sec', 'mA', 
                            'endpoint', 'grip/rotation', 'torque', 'force', 'ori', '#']

        df.columns = expected_columns
        df = df[df['r'] != 's']

        # 필요하지 않은 칼럼 삭제
        df = df.drop(['r', 'grip/rotation', '#'], axis=1)

        # endpoint 및 ori 처리
        df[['x_end', 'y_end', 'z_end']] = df['endpoint'].str.split('/', expand=True)
        df[['yaw', 'pitch', 'roll']] = df['ori'].str.split('/', expand=True)

        for col in ['deg', 'deg/sec', 'torque', 'force']:
            parts = df[col].str.split('/', expand=True)
            parts.columns = [f'{col}{i+1}' for i in range(parts.shape[1])]
            df = pd.concat([df, parts], axis=1)

        # 원본 칼럼 삭제
        df = df.drop(['deg', 'deg/sec', 'mA', 'endpoint', 'torque', 'force', 'ori'], axis=1)

        # 모든 칼럼을 숫자형으로 변환
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        df['deg2'] = df['deg2'] - 90
        df['deg4'] = df['deg4'] - 90

        # 'time' 칼럼 생성 및 데이터 정렬
        df['time'] = df['timestamp'] - df['sequence'] - 1
        df = df.drop(['sequence', 'timestamp'], axis=1)
        df = df.sort_values(by=["time"], ascending=True).reset_index(drop=True)

        return df.values
    
    def get_random_trajectory_by_type(self, folder_path, movement_type):
        if not os.path.exists(folder_path):
            raise ValueError(f"경로가 존재하지 않습니다: {folder_path}")
            
        trajectory_files = [f for f in os.listdir(folder_path) 
                            if f.endswith('.txt') and f.startswith(movement_type)]
        
        if not trajectory_files:
            raise ValueError(f"폴더에 {movement_type} 타입의 궤적 파일이 없습니다: {folder_path}")
            
        selected_file = random.choice(trajectory_files)
        file_path = os.path.join(folder_path, selected_file)
        
        df = pd.read_csv(file_path, delimiter=',')
        processed_df = self.process_data(df)

        return processed_df, selected_file

    def get_available_movement_types(self):
        golden_files = os.listdir(self.golden_dir)
        movement_types = set()
        
        for file in golden_files:
            if file.endswith('.txt'):
                movement_type = '_'.join(file.split('_')[:2])

                movement_types.add(movement_type)
        
        return sorted(list(movement_types))

    def load_data(self):
        all_data = []
        
        print("Loading golden samples...")
        for file in os.listdir(self.golden_dir):
            if file.endswith('.txt'):
                file_path = os.path.join(self.golden_dir, file)
                try:
                    df = pd.read_csv(file_path, delimiter=',')
                    processed_df = self.process_data(df)
                    all_data.append((processed_df, 1))  
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

        print("Loading non-golden samples...")
        for file in os.listdir(self.non_golden_dir):
            if file.endswith('.txt'):
                file_path = os.path.join(self.non_golden_dir, file)
                try:
                    df = pd.read_csv(file_path, delimiter=',')
                    processed_df = self.process_data(df)
                    all_data.append((processed_df, 0))
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
                
        data, labels = zip(*all_data)
        
        for i, d in enumerate(data):
            if d.shape[1] != 21:  
                print(f"Inconsistent shape at index {i}: {d.shape}")
        
        data = np.vstack(data)
        self.scaler.fit(data)
        
        for d, label in zip(data, labels):
            scaled_data = self.scaler.transform([d])[0]
            self.data.append(torch.tensor(scaled_data, dtype=torch.float32))
            self.labels.append(label)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def set_model(self, model):
        """학습된 모델 설정"""
        self.model = model
        return self

    def predict_random_non_golden_trajectory(self):

        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")

        non_golden_files = [f for f in os.listdir(self.non_golden_dir) if f.endswith('.txt')]
        selected_file = random.choice(non_golden_files)
        file_path = os.path.join(self.non_golden_dir, selected_file)
        
        try:
            # 데이터 로드 및 전처리
            df = pd.read_csv(file_path, delimiter=',')
            processed_df = self.process_data(df)  # 이미 21개의 열을 가진 형태로 처리됨
            scaled_data = self.scaler.transform(processed_df)

            data_tensor = torch.FloatTensor(scaled_data).unsqueeze(0).to(device)
            
            # 텐서로 변환 (배치 차원 추가)
            # [시퀀스 길이, 특성 수] -> [배치 크기=1, 시퀀스 길이, 특성 수]
            data_tensor = torch.FloatTensor(scaled_data).unsqueeze(0).to(device)

            # 모델로 예측
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(data_tensor)
                probabilities = torch.softmax(outputs, dim=1)  
                predicted = torch.argmax(probabilities, dim=1)
                predicted_type = self.classes[int(predicted[0].item())] 
                predicted_prob = probabilities[0][predicted[0]].item()
            
            print(f"Selected non-golden trajectory: {selected_file}")
            print(f"Predicted movement type: {predicted_type}")
            
            class_names = self.classes  # 클래스 이름이 저장된 리스트
            prob_values = probabilities[0].cpu().numpy()  # 예측된 확률 값 (CPU로 이동)

            # 막대 그래프 시각화
            plt.figure(figsize=(10, 6))
            plt.bar(class_names, prob_values, color='skyblue')
            plt.xlabel('Classes')
            plt.ylabel('Probability')
            plt.title('Class Prediction Probabilities')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()

            return selected_file, predicted_type, processed_df
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None, None, None

def collate_fn(batch):
        data = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        padded_data = []
        for d in data:
            if len(d.shape) == 1:  
                d = d.unsqueeze(0)  
                
            if d.shape[0] > 21:
                padded_data.append(d[:21])
            else:
                padding_length = 21 - d.shape[0]
                padding = torch.zeros(padding_length, d.shape[1])
                padded = torch.cat([d, padding], dim=0)
                padded_data.append(padded)
        
        data = torch.stack(padded_data)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return data, labels

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes, max_len=2000):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  
            dropout=0.1,
            batch_first=True,
            activation='gelu' 
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model * 2, num_classes)
        )
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x)
        
        positional_encoding = self.positional_encoding[:, :seq_len, :] 
        x = x + positional_encoding
        
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) 
        return self.fc(x)

def train_classification_model(dataset, unique_labels, model_params):
    input_dim = model_params.get('input_dim', 21)
    d_model = model_params.get('d_model', 128)
    nhead = model_params.get('nhead', 2)
    num_layers = model_params.get('num_layers', 5)
    num_classes = len(unique_labels)  
    batch_size = model_params.get('batch_size', 64)
    epochs = model_params.get('epochs', 100)

    model = TransformerModel(input_dim, d_model, nhead, num_layers, num_classes).to(device)


    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=False, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0
    train_losses = []
    val_accuracies = []
    
    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(initialize_weights)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        # 검증
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        val_acc = val_correct / val_total
        val_accuracies.append(val_acc)

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Accuracy: {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_classification_model.pth')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.show()
    
    return model, dataset.scaler

def test_classification_model(dataset, unique_labels, model_params):
    input_dim = model_params.get('input_dim', 21)
    d_model = model_params.get('d_model', 128)
    nhead = model_params.get('nhead', 2)
    num_layers = model_params.get('num_layers', )
    num_classes = len(unique_labels) 
    batch_size = model_params.get('batch_size', 64)
    epochs = model_params.get('epochs', 100)
    
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = TransformerModel(input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, num_classes=num_classes).to(device)
    model_weights = torch.load('best_classification_model.pth', weights_only=True)
    model.load_state_dict(model_weights, strict=True)
    model.eval()  

    test_correct = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            test_correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_accuracy = test_correct / len(dataset)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return model, dataset.scaler
    
if __name__ == "__main__":
    # set_seed()
    
    base_path = "data" 
    dataset = TrajectoryDataset(base_path)
    
    model_params = {
        'input_dim': 21,  
        'd_model': 64,    
        'nhead': 2,       
        'num_layers': 3,  
        'batch_size': 64, 
        'epochs': 100     
    }
    
    print("Training Start...")
    model, scaler = train_classification_model(
        dataset, 
        dataset.classes,  
        model_params
    )
    dataset.set_model(model) 
    print("Training complete.")

    print("\nEvaluation Start...")
    model, scaler = test_classification_model(
        dataset, 
        dataset.classes,  
        model_params
    )

    print("Evaluation complete.")

    print("\nPredicting random trajectory...")
    selected_file, predicted_type, trajectory = dataset.predict_random_non_golden_trajectory()

