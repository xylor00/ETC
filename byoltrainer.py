import torch
from byol_pytorch_fix import BYOL
from traffic_augmentation import TrafficAugmentation, IdentityAugmentation
import pandas as pd
from GRUnet import GRUBackbone
from torch.utils.data import Dataset, DataLoader
import numpy as np
import multiprocessing


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
categories = ["Email", "Chat", "Streaming", "File Transfer", "VoIP", "P2P"]

# 自定义数据集类 --------------------------------------------------
class CSVDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        # 读取CSV数据
        self.data = pd.read_csv(csv_path, skiprows=1, header=None)
        
        # 分离特征和标签
        self.features = self.data.iloc[:, :-1].values.astype('float32')
        self.labels = self.data.iloc[:, -1].values
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = categories.index(self.labels[idx])
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
if __name__ == '__main__':
    multiprocessing.freeze_support()
    # 模拟拆包粘包增强
    aug_1 = TrafficAugmentation()
    aug_2 = IdentityAugmentation()

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

    num_epochs = 10
    batch_size=100
    train_dataset = CSVDataset('features/flow_sequences.csv')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
    total_step = len(train_loader) 

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    for epoch in range(num_epochs):
        for i, (flows, labels) in enumerate(train_loader):
            
            flows = flows.to(device)     # 输入数据移至GPU
            labels = labels.to(device)   # 标签移至GPU（如果计算损失需要）
            # 前向传播
            loss = learner(flows)
            
            # 反向传播和优化
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            learner.update_moving_average() # 目标网络参数指数移动平均更新
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item()}')  # 输出损失值
                
    # 确保模型切换到评估模式（关闭训练专用层）
    gru_backbone.eval()

    # 创建用于特征提取的数据加载器（不打乱顺序）
    feature_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)

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

    """
    # 存储模型
    torch.save(gru_backbone.state_dict(), 'model/improved-net.pt')
    """