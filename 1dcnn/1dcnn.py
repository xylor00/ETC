import torch
import torch.nn as nn
import torch.nn.functional as F # 导入 F 用于激活函数和池化
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import multiprocessing
import os # 导入 os 模块用于路径操作

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#categories = ["socialapp", "chat", "email", "file", "streaming", "VoIP"]
#categories = ["socialapp", "chat", "email", "file", "streaming", "web"]
categories = ["Benign", "Malware"]

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
    
    return matrix.numpy()
    
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
        # 获取当前特征，一个长度为 784 的一维向量
        feature_vector = self.features[idx]
        
        # 将 784 字节的向量重塑为 28x28 的灰度图像
        image_feature = feature_vector.reshape(1, 28, 28)

        return (
            torch.tensor(image_feature, dtype=torch.float32),
            torch.tensor(self.labels[idx]).long()
        )

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()


        # 输入通道为 1 (灰度图)，输出 32 通道
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二个卷积层
        # 输入通道为 32，输出 64 通道
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 第二个池化层
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1_input_features = 64 * 7 * 7 

        self.fc1 = nn.Linear(self.fc1_input_features, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):

        # 第一个卷积层和池化层
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # 第二个卷积层和池化层
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(-1, self.fc1_input_features)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # 输出层 (logits)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 超参数
    # input_size 不再直接用于模型初始化，但可以保留作为数据特征长度的提示
    input_size = 784  # 对应CSV中的原始流字节数量
    num_classes = len(categories)
    num_epochs = 100
    batch_size = 512
    lr = 0.02

    # 定义模型保存目录
    model_dir = '1dcnn'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 加载原始数据
    full_raw = RawDataset(os.path.join(model_dir, '784byte.csv'))

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

    model = SimpleCNN(num_classes).to(device) 

    criterion = nn.CrossEntropyLoss()

    # 优化器为Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 早停参数
    best_avg_val_loss = 100
    patience = 3
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
        if avg_val_loss < best_avg_val_loss:
            best_avg_val_loss = avg_val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(model_dir, 'cnn_best_model.pth'))
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                stop_training = True
        
        # 打印信息
        print(f"Epoch {epoch+1}: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, "
              f"Val Acc={val_acc:.2%}", flush=True)
        
    # 测试评估
    model.load_state_dict(torch.load(os.path.join(model_dir, 'cnn_best_model.pth')))
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
    # 避免除以零的警告
    normalized_conf = np.zeros_like(conf_matrix, dtype=float)
    row_sums = conf_matrix.sum(axis=1)
    for i in range(num_classes):
        if row_sums[i] > 0:
            normalized_conf[i, :] = conf_matrix[i, :] / row_sums[i]

    # 输出带百分比的矩阵
    print("\nNormalized Confusion Matrix (row-wise):")
    norm_df = pd.DataFrame(
        normalized_conf,
        index=categories,
        columns=categories
    ).applymap(lambda x: f"{x:.1%}") # 转换为百分比格式

    print(norm_df)
    norm_df.to_csv(os.path.join(model_dir, 'cnn_confusion_matrix.csv'), index_label='True\Predicted')