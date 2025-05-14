import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import multiprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
categories = ["Email", "Chat", "Streaming", "File Transfer", "VoIP"]

# 数据集类（含标准化）
class CSVDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path, skiprows=1, header=None)
        features = self.data.iloc[:, :-1].values.astype('float32')
        labels = self.data.iloc[:, -1].values
        
        # 计算训练集的均值和标准差（实际使用时应仅在训练集计算）
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        self.features = (features - self.mean) / (self.std + 1e-8)
        
        self.labels = [categories.index(lbl) for lbl in labels]  # 提前转换标签

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

# 改进的模型（添加正则化）
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # 超参数
    input_size = 256  
    hidden_size = 500
    num_classes = len(categories)
    num_epochs = 100
    batch_size = 100
    lr = 0.001

    # 数据加载
    full_dataset = CSVDataset('features/merge_features.csv')
    train_size = int(0.6 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)


    # 初始化
    model = MLP(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)  # 动态学习率

    # 早停参数
    best_avg_val_loss = 100
    patience = 5
    no_improve_epochs = 0
    stop_training = False
    
    for epoch in range(num_epochs):
        # 检查早停条件
        if stop_training:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # 训练阶段
        model.train()
        train_loss = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            
        # 计算训练损失    
        avg_train_loss = np.mean(train_loss)
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                val_loss += criterion(preds, y).item()
                val_correct += (preds.argmax(1) == y).sum().item()
        val_acc = val_correct / len(val_dataset)
        avg_val_loss = val_loss / len(val_loader)
        
        # 早停判断
        if avg_val_loss < best_avg_val_loss:
            best_avg_val_loss = avg_val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'model/best_model.pth')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                stop_training = True
        
        scheduler.step(avg_val_loss)  # 调整学习率
        
        # 打印信息
        print(f"Epoch {epoch+1}: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, "
              f"Val Acc={val_acc:.2%}")
        
    # 测试评估
    model.load_state_dict(torch.load('model/best_model.pth'))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    
    print("\nTest Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=categories))