import torch
from byol_pytorch_fix import BYOL
from traffic_augmentation import TrafficAugmentation, IdentityAugmentation
import pandas as pd
from GRUnet import GRUBackbone
from torch.utils.data import Dataset, DataLoader
import numpy as np
import multiprocessing
from torch.utils.data import random_split



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
    

    
if __name__ == '__main__':
    multiprocessing.freeze_support()
    # 模拟拆包粘包增强
    aug_1 = TrafficAugmentation()
    aug_2 = IdentityAugmentation()
    
    # 超参数
    num_epochs = 50
    batch_size=100

    # 划分训练集和验证集（8:2比例）
    full_dataset = CSVDataset('features/flow_sequences.csv')
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)


    # 初始化GRU骨干网络
    gru_backbone = GRUBackbone(
        input_dim=1,          # 假设输入是单变量时间序列
        hidden_dim=128,
        output_dim=256        # 需与projection_size一致
    ).to(device)

    # 创建BYOL实例
    learner = BYOL(
        net=gru_backbone,
        input_dim=100,
        augment_fn=aug_1,
        augment_fn2=aug_2,
        projection_size=256,
        projection_hidden_size=512
    ).to(device)

    # 创建优化器
    optimizer = torch.optim.Adam(learner.parameters(), lr=0.001)
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
            
        scheduler.step(avg_val_loss)  # 调整学习率
        
        # 打印信息
        print(f"Epoch {epoch+1}: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, "
              f"Best Val Loss={best_avg_val_loss:.4f}")
            
                
    # 确保模型切换到评估模式用于处理数据（关闭训练专用层）
    gru_backbone.load_state_dict(torch.load('model/improved-net.pth'))
    gru_backbone.eval()

    # 创建用于特征提取的数据加载器（不打乱顺序）
    feature_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)

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