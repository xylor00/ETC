import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import multiprocessing
from torch.optim.lr_scheduler import OneCycleLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
categories = ["socialapp", "chat", "email", "file", "streaming", "VoIP"]
#categories = ["socialapp", "chat", "email", "file", "streaming", "web"]
#categories = ["Benign", "Malware"]
#categories = ["chat", "file", "streaming", "VoIP", "C2"]

def compute_confusion_matrix(true_labels, pred_labels, num_classes):
        """
        使用PyTorch快速计算混淆矩阵
        返回形状为 [num_classes, num_classes] 的矩阵
        """
        # 转换数据为LongTensor
        true_labels = torch.as_tensor(true_labels, dtype=torch.long)
        pred_labels = torch.as_tensor(pred_labels, dtype=torch.long)
        
        # 确保标签在有效范围内
        assert torch.all(true_labels >= 0) & torch.all(true_labels < num_classes)
        assert torch.all(pred_labels >= 0) & torch.all(pred_labels < num_classes)
        
        # 计算线性索引
        indices = true_labels * num_classes + pred_labels
        
        # 使用bincount统计频次
        matrix = torch.bincount(
            indices,
            minlength=num_classes**2
        ).reshape(num_classes, num_classes)
        
        return matrix.numpy()  # 转换为numpy数组方便后续处理
    
# 数据集类（不含标准化）    
class RawDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path, skiprows=1, header=None)
        self.features = self.data.iloc[:, :-1].values.astype('float32')
        self.labels = self.data.iloc[:, -1].values

    def __len__(self): 
        return len(self.data)
    def __getitem__(self, idx): 
        return self.features[idx], self.labels[idx]

# 数据集类（含标准化） 
class StandardizedDataset(Dataset):
    def __init__(self, raw_dataset, indices, mean, std):
        self.features = (raw_dataset.features[indices] - mean) / (std + 1e-8)
        self.labels = [categories.index(raw_dataset.labels[i]) for i in indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx]),
            torch.tensor(self.labels[idx])
        )

# 定义残差块 (Residual Block)
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.2):
        super().__init__()
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.dropout = nn.Dropout(dropout_rate) 
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out) 
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out += residual 
        out = self.relu(out) 
        
        return out
    
# MLP模型 (添加残差连接)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=3):
        super().__init__()
        
        # 初始线性层
        self.initial_fc = nn.Linear(input_size, hidden_size)
        self.initial_bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        
        # 核心残差块堆叠
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ResidualBlock(hidden_size))
            
        # 最后的分类层
        self.final_fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # 初始处理
        x = self.initial_fc(x)
        x = self.initial_bn(x)
        x = self.relu(x)

        # 核心残差块处理
        for layer in self.layers:
            x = layer(x)
        
        # 输出分类 logits
        return self.final_fc(x)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # 超参数
    input_size = 256  
    hidden_size = 1024
    MLPlayers = 5
    num_classes = len(categories)
    num_epochs = 5000
    batch_size = 512
    lr = 1e-4

    # 加载原始数据
    full_raw = RawDataset('features/flevel_features.csv')
    
    train_size = int(0.6 * len(full_raw))
    val_size = int(0.2 * len(full_raw))
    test_size = len(full_raw) - train_size - val_size

    # 分割数据集
    train_subset, val_subset, test_subset = random_split(
        full_raw, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(37)
    )
    
    # 获取各子集的索引
    train_indices = train_subset.indices
    val_indices = val_subset.indices
    test_indices = test_subset.indices

    # 计算训练集的均值和标准差（仅使用训练数据）
    train_features = full_raw.features[train_indices]
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)

    # 创建标准化数据集
    train_dataset = StandardizedDataset(full_raw, train_indices, mean, std)
    val_dataset = StandardizedDataset(full_raw, val_indices, mean, std)
    test_dataset = StandardizedDataset(full_raw, test_indices, mean, std)
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)


    # 初始化
    model = MLP(input_size, hidden_size, num_classes, num_layers=MLPlayers).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # 优化器为AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,                  # 设置基础学习率
        weight_decay=1e-3,
        betas=(0.9, 0.98)
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',      # 监测 acc 增长
        factor=0.5,      # 触发后 LR 减半
        patience=150    # 200个epoch没提升就降速
    )

    # 早停参数
    best_avg_val_loss = 100
    best_acc = 0.0
    decay_patience = 200  # 如果200代没提升，就回溯并降温
    patience = 1000      # 如果1000代没提升，就停止训练
    no_improve_epochs = 0
    stop_training = False
    
    for epoch in range(num_epochs):
        # 检查早停条件
        if stop_training:
            print(f"Early stopping at epoch {epoch}", flush=True)
            print(f"get model at epoch {epoch-patience}", flush=True)
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
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'model/best_model.pth')
        else:
            no_improve_epochs += 1
            if no_improve_epochs % decay_patience == 0 and no_improve_epochs > 0:
                print(f"Detected jitter. Loading best model and reducing LR...", flush=True)
                model.load_state_dict(torch.load('model/best_model.pth'))
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
            if no_improve_epochs >= patience:
                stop_training = True
        
        scheduler.step(val_acc)  # 执行主调度
        
        # 打印信息
        print(f"Epoch {epoch+1}: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, "
              f"Val Acc={val_acc:.2%}, "
              f"best Acc={best_acc:.2%}", flush=True)
        
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
            
    
    
    print("\nTest Classification Report:", flush=True)
    print(classification_report(all_labels, all_preds, target_names=categories, digits=4), flush=True)
    
    
    # 混淆矩阵输出
    conf_matrix = compute_confusion_matrix(all_labels, all_preds, num_classes=len(categories))

   # 计算归一化混淆矩阵（按行）
    normalized_conf = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # 输出带百分比的矩阵
    print("\nNormalized Confusion Matrix (row-wise):")
    norm_df = pd.DataFrame(
        normalized_conf,
        index=categories,
        columns=categories
    ).applymap(lambda x: f"{x:.1%}")  # 转换为百分比格式

    print(norm_df)
    norm_df.to_csv('model/confusion_matrix.csv', index_label='True\Predicted')