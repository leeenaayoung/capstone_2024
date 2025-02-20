import csv
import torch
import os
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
from utils import preprocess_trajectory_data

##########################
# 데이터셋 로드
##########################
class ClassificationDataset(Dataset):
    def __init__(self, base_path):
        self.data_cache = {}
        self.data = []
        self.labels = []
        self.scaler = StandardScaler()
        self.load_data(base_path)
        self.unique_labels = sorted(list(set(self.labels))) 

        # 모든 데이터를 모은 후, 스케일링 적용
        all_data = torch.cat(self.data, dim=0)  # (row-wise) 통합
        all_data_np = all_data.numpy()
        self.scaler.fit(all_data_np)  # 스케일러에 fit

        # 스케일링 후 다시 저장
        scaled_data_list = []
        for sample in self.data:
            sample_np = sample.numpy()
            sample_scaled = self.scaler.transform(sample_np)
            scaled_data_list.append(torch.tensor(sample_scaled, dtype=torch.float32))

        self.data = scaled_data_list  # 최종 교체

    def load_data(self, base_path):
        """txt 파일명에서 두 번째 언더바 전까지를 라벨로 사용"""
        for file_name in sorted(os.listdir(base_path)):
            if file_name.endswith('.txt'):
                # 파일명에서 라벨 추출 (두 번째 언더바 전까지)
                label = '_'.join(file_name.split('_')[:2])  
                
                file_path = os.path.join(base_path, file_name)
                with open(file_path, 'r') as f:
                    data_list = list(csv.reader(f))
                
                # 데이터 전처리
                data_v = preprocess_trajectory_data(data_list)
                tensor_data = torch.tensor(data_v.values, dtype=torch.float32)

                self.data.append(tensor_data)
                self.labels.append(label)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
# 라벨 목록
def get_unique_labels(base_path):
    tmp_dataset = ClassificationDataset(base_path)
    return sorted(list(set(tmp_dataset.labels)))

# 기존
def collate_fn(batch, dataset):
    global unique_labels  
    data_list = [item[0] for item in batch]
    label_list = [item[1] for item in batch]

    data_padded = pad_sequence(data_list, batch_first=True, padding_value=0)
    label_indices = [dataset.unique_labels.index(lbl) for lbl in label_list]
    label_indices = torch.tensor(label_indices, dtype=torch.long)

    return data_padded, label_indices

def classification_dataloaders(base_path, batch_size, train_ratio=0.8, val_ratio=0.1):
    """분류 모델 데이터로더"""
    dataset = ClassificationDataset(base_path)

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    from functools import partial
    collate_with_dataset = partial(collate_fn, dataset=dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            collate_fn=collate_with_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          collate_fn=collate_with_dataset, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           collate_fn=collate_with_dataset, shuffle=False)
    
    return train_loader, val_loader, test_loader
