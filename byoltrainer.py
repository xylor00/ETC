import torch
from byol_pytorch_fix import BYOL
from traffic_augmentation import TrafficAugmentation, IdentityAugmentation
import pandas as pd
from GRUnet import GRUBackbone
from torch.utils.data import Dataset, DataLoader
import numpy as np
import multiprocessing
from torch.utils.data import random_split
from torch.optim.lr_scheduler import OneCycleLR



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
categories = ["socialapp", "chat", "email", "file", "streaming", "VoIP"]
#categories = ["socialapp", "chat", "email", "file", "streaming", "web"]
#categories = ["Benign", "Malware"]
#categories = ["chat", "file", "streaming", "VoIP", "C2"]

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

# 分割数据集类（含标准化） 
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

#完整数据集类        
class FullDataset(Dataset):
    def __init__(self, raw_dataset, mean, std):
        self.raw_dataset = raw_dataset
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.raw_dataset)
        
    def __getitem__(self, idx):
        x = (self.raw_dataset.features[idx] - self.mean) / (self.std + 1e-8)
        y = categories.index(self.raw_dataset.labels[idx])
        return torch.tensor(x), torch.tensor(y)
    
if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # 超参数
    num_epochs = 100
    batch_size=256
    lr = 2e-5

    # 加载原始数据
    full_raw = RawDataset('features/flow_sequences.csv')
    
    train_size = int(0.8 * len(full_raw))
    val_size = len(full_raw) - train_size

    # 分割数据集
    train_subset, val_subset = random_split(
        full_raw, [train_size, val_size],
        generator=torch.Generator().manual_seed(37)
    )
    
    # 获取各子集的索引
    train_indices = train_subset.indices
    val_indices = val_subset.indices

    # 计算训练集的均值和标准差（仅使用训练数据）
    train_features = full_raw.features[train_indices]
    mean = train_features.mean()        # 标量（全局均值）
    std = train_features.std()          # 标量（全局标准差）

    # 创建标准化数据集
    train_dataset = StandardizedDataset(full_raw, train_indices, mean, std)
    val_dataset = StandardizedDataset(full_raw, val_indices, mean, std)
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)
    
    # 创建用于特征提取的数据加载器（不打乱顺序）
    full_dataset = FullDataset(full_raw, mean, std)
    feature_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)

    # 初始化GRU骨干网络
    gru_backbone = GRUBackbone(
        input_dim=1,          
        hidden_dim=64,
        output_dim=128        
    ).to(device)

    # 模拟拆包粘包增强
    aug_1 = TrafficAugmentation(mean=mean.item(), std=std.item())
    aug_2 = IdentityAugmentation()
    
    # 创建BYOL实例
    learner = BYOL(
        net=gru_backbone,
        input_dim=100,
        augment_fn=aug_1,
        augment_fn2=aug_2,
        projection_size=128,
        projection_hidden_size=256,
        moving_average_decay=0.996  # EMA动量（默认0.99可能过低）
    ).to(device)

    optimizer = torch.optim.AdamW(
        learner.parameters(),
        lr=lr,                  # 基础学习率
        weight_decay=1e-4,
        betas=(0.9, 0.98)
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr*15,      # 峰值学习率
        total_steps=num_epochs,
        pct_start=0.1,     # warmup阶段
        anneal_strategy='cos',
        div_factor=25,     # 初始学习率与峰值比率
        final_div_factor=1e4
    )
    
    # 早停参数
    best_avg_val_loss = 100
    patience = 10
    no_improve_epochs = 0
    stop_training = False
    
    for epoch in range(num_epochs):
        # 检查早停条件
        if stop_training:
            print(f"Early stopping at epoch {epoch}", flush=True)
            print(f"get model at epoch {epoch-patience}", flush=True)
            break
        
        # 训练阶段
        learner.train()
        train_loss = []
        for flows, labels in train_loader:
            
            flows = flows.to(device)     # 输入数据移至GPU            
            
            # 前向传播
            loss = learner(flows)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=1.0)  # 限制梯度范数
            learner.update_moving_average() # 目标网络参数指数移动平均更新
            
            train_loss.append(loss.item())
            
        # 计算训练损失    
        avg_train_loss = np.mean(train_loss)
            
        # 验证阶段
        learner.eval()
        val_loss = 0.0
                
        with torch.no_grad():
            for flows_val, _ in val_loader:  # 自监督不需要标签
                flows_val = flows_val.to(device)
                                
                val_loss += learner(flows_val)  # BYOL的前向计算
                        
        avg_val_loss = val_loss / len(val_loader)
        
        # 早停判断
        if avg_val_loss < best_avg_val_loss:
            best_avg_val_loss = avg_val_loss
            no_improve_epochs = 0
            # 存储模型
            torch.save(gru_backbone.state_dict(), 'model/improved-net.pth')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                stop_training = True            
            
        scheduler.step()  # 执行调度
            
        # 打印信息
        print(f"Epoch {epoch+1}: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, "
              f"Best Val Loss={best_avg_val_loss:.4f}", flush=True)
            
                
    # 确保模型切换到评估模式用于处理数据（关闭训练专用层）
    gru_backbone.load_state_dict(torch.load('model/improved-net.pth'))
    gru_backbone.eval()

    # 存储所有特征和标签的容器
    all_features = []
    all_labels = []

    # 禁用梯度计算以提升效率
    with torch.no_grad():
        for flows, labels in feature_loader:
            # 调整输入形状为GRU期望的格式
            # 输入形状应为 [batch_size, 序列长度, 特征维度]
            flows = flows.unsqueeze(-1).to(device)  # 从 [50,100] -> [50,100,1]
            
            # 通过GRU骨干网络获取高级特征
            features = gru_backbone(flows)  # 输出形状应为 [50,256]
            
            # 收集结果（转移到CPU并转numpy）
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 合并所有batch的结果

    all_features = np.concatenate(all_features, axis=0)  # 形状 [总样本数,256]
    all_labels = np.concatenate(all_labels, axis=0)       # 形状 [总样本数]

    # 创建包含特征和标签的DataFrame
    # 生成特征列名：feature_0, feature_1...feature_255
    feature_columns = [f'feature_{i}' for i in range(all_features.shape[1])]
    df = pd.DataFrame(all_features, columns=feature_columns)

    # 添加标签列（转换为原始字符串标签）
    df['label'] = pd.Series(all_labels).map(lambda x: categories[x])

    # 保存到CSV文件
    df.to_csv('features/flevel_features.csv', index=False)