from models import ClassificationDataset, GenerationDataset
from torch.utils.data import DataLoader, random_split

def classification_dataloaders(base_path, batch_size, collate_fn = True, train_ratio=0.8, val_ratio=0.1):
    """분류 모델용 데이터로더 생성"""
    dataset = ClassificationDataset(base_path)

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn = True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn = True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn = True, shuffle=False)
    
    return train_loader, val_loader, test_loader

def generation_dataloaders(base_path, batch_size, train_ratio=0.8, val_ratio=0.1):
    """생성 모델용 데이터로더 생성"""
    dataset = GenerationDataset(base_path)

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader