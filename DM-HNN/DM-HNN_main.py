import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import multiprocessing
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
categories = ["socialapp", "chat", "email", "file", "streaming", "VoIP"]

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

# 数据集类（同时加载流级和包级特征）
class DualFeatureDataset(Dataset):
    def __init__(self, flow_csv, pkt_csv):
        # 读取流级特征CSV
        flow_df = pd.read_csv(flow_csv)
        self.flow_features = flow_df.iloc[:, :-1].values.astype('float32')
        self.labels = flow_df.iloc[:, -1].values
        
        # 读取包级特征CSV
        pkt_df = pd.read_csv(pkt_csv)
        self.pkt_features = pkt_df.iloc[:, :-1].values.astype('float32')
        
        # 确保数据一致性
        assert len(self.flow_features) == len(self.pkt_features)
        assert len(self.flow_features) == len(self.labels)
        
        # 标签编码
        self.label_map = {label: idx for idx, label in enumerate(np.unique(self.labels))}
        self.encoded_labels = np.array([self.label_map[label] for label in self.labels])
        self.reverse_label_map = {idx: label for label, idx in self.label_map.items()}

    def __len__(self):
        return len(self.flow_features)

    def __getitem__(self, idx):
        return (
            self.flow_features[idx],
            self.pkt_features[idx],
            self.encoded_labels[idx]
        )

# 标准化数据集类
class StandardizedDualFeatureDataset(Dataset):
    def __init__(self, raw_dataset, indices):
        # 提取原始特征
        self.flow_features = raw_dataset.flow_features[indices]
        self.pkt_features = raw_dataset.pkt_features[indices]
        self.labels = raw_dataset.encoded_labels[indices]
        
        # 计算训练集的均值和标准差（仅使用训练数据）
        if not hasattr(self, 'flow_mean'):
            # 计算流级特征的均值和标准差
            self.flow_mean = self.flow_features.mean(axis=0)
            self.flow_std = self.flow_features.std(axis=0)
            
            # 计算包级特征的均值和标准差
            self.pkt_mean = self.pkt_features.mean(axis=0)
            self.pkt_std = self.pkt_features.std(axis=0)
        
        # 标准化特征
        self.flow_features = (self.flow_features - self.flow_mean) / (self.flow_std + 1e-8)
        self.pkt_features = (self.pkt_features - self.pkt_mean) / (self.pkt_std + 1e-8)
        
        # 转换为PyTorch张量
        self.flow_features = self.flow_features.astype('float32')
        self.pkt_features = self.pkt_features.astype('float32')
        self.labels = self.labels.astype('int64')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.flow_features[idx]),
            torch.tensor(self.pkt_features[idx]),
            torch.tensor(self.labels[idx])
        )

# 修正后的门控激活机制 (论文中的特征融合)
class GatedActivationFusion(nn.Module):
    def __init__(self, flow_dim, pkt_dim, hidden_dim):
        super(GatedActivationFusion, self).__init__()
        
        # 预融合层 (拼接两个特征)
        self.prefusion_fc = nn.Linear(flow_dim + pkt_dim, hidden_dim)
        self.prefusion_activation = nn.Tanh()
        
        # 门控参数
        self.alpha_flow = nn.Parameter(torch.tensor(0.1))  # 可学习参数α_flow
        self.alpha_pkt = nn.Parameter(torch.tensor(0.1))   # 可学习参数α_pkt
        
        # 相似度计算层 - 修正为正确的输入维度
        self.similarity_fc_flow = nn.Linear(flow_dim, 1, bias=False)  # 流特征维度
        self.similarity_fc_pkt = nn.Linear(pkt_dim, 1, bias=False)    # 包特征维度
        
        # 最终融合层
        self.fusion_fc = nn.Linear(flow_dim + pkt_dim, hidden_dim)
        
    def forward(self, flow_feature, pkt_feature):
        # 拼接两个特征
        combined = torch.cat([flow_feature, pkt_feature], dim=1)
        
        # 预融合层 (论文中的F)
        F = self.prefusion_activation(self.prefusion_fc(combined))
        
        # 计算每个特征对决策特征F的影响力因子 (论文公式15)
        # 计算流级特征的影响力
        similarity_flow = self.similarity_fc_flow(flow_feature)
        
        # 计算包级特征的影响力
        similarity_pkt = self.similarity_fc_pkt(pkt_feature)
        
        # 归一化得到影响力因子
        exp_sum = torch.exp(similarity_flow) + torch.exp(similarity_pkt)
        I_flow = torch.exp(similarity_flow) / (exp_sum + 1e-10)
        I_pkt = torch.exp(similarity_pkt) / (exp_sum + 1e-10)
        
        # 调整特征权重 (论文公式16)
        # 添加epsilon避免log(0)
        epsilon = 1e-10
        flow_adjusted = flow_feature * (1 + self.alpha_flow * torch.log(I_flow + epsilon))
        pkt_adjusted = pkt_feature * (1 + self.alpha_pkt * torch.log(I_pkt + epsilon))
        
        # 拼接调整后的特征
        adjusted_combined = torch.cat([flow_adjusted, pkt_adjusted], dim=1)
        
        # 最终融合
        fused = self.fusion_fc(adjusted_combined)
        return fused

# 完整分类模型
class DMHNNClassifier(nn.Module):
    def __init__(self, flow_dim, pkt_dim, hidden_dim, num_classes):
        super(DMHNNClassifier, self).__init__()
        
        # 特征融合模块
        self.fusion = GatedActivationFusion(flow_dim, pkt_dim, hidden_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, flow_feature, pkt_feature):
        # 特征融合
        fused = self.fusion(flow_feature, pkt_feature)
        
        # 分类
        output = self.classifier(fused)
        return output

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # 确保模型目录存在
    os.makedirs('model', exist_ok=True)
    
    # 超参数
    flow_dim = 256  # 流级特征维度
    pkt_dim = 256   # 包级特征维度
    hidden_dim = 500  
    num_classes = len(categories)
    num_epochs = 200
    batch_size = 512
    lr = 0.01

    # 加载原始数据
    full_raw = DualFeatureDataset(
        flow_csv='DM-HNN/final_flevel_features.csv',
        pkt_csv='DM-HNN/final_plevel_features.csv'
    )
    
    # 分割数据集
    train_size = int(0.6 * len(full_raw))
    val_size = int(0.2 * len(full_raw))
    test_size = len(full_raw) - train_size - val_size
    
    train_indices, val_indices, test_indices = random_split(
        range(len(full_raw)), [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(37)
    )
    
    # 创建标准化数据集
    train_dataset = StandardizedDualFeatureDataset(full_raw, train_indices)
    val_dataset = StandardizedDualFeatureDataset(full_raw, val_indices)
    test_dataset = StandardizedDualFeatureDataset(full_raw, test_indices)
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             pin_memory=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           pin_memory=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            pin_memory=True, num_workers=4, persistent_workers=True)

    # 初始化模型
    model = DMHNNClassifier(
        flow_dim=flow_dim,
        pkt_dim=pkt_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # 优化器为AdamW
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,                  # 设置基础学习率
    )

    # 早停参数
    best_avg_val_loss = 100
    patience = 10
    no_improve_epochs = 0
    stop_training = False
    
    # 训练循环
    for epoch in range(num_epochs):
        # 检查早停条件
        if stop_training:
            print(f"Early stopping at epoch {epoch}", flush=True)
            print(f"get model at epoch {epoch-patience}", flush=True)
            break
        
        # 训练阶段
        model.train()
        train_loss = []
        for flow_x, pkt_x, y in train_loader:
            flow_x, pkt_x, y = flow_x.to(device), pkt_x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(flow_x, pkt_x)
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
            for flow_x, pkt_x, y in val_loader:
                flow_x, pkt_x, y = flow_x.to(device), pkt_x.to(device), y.to(device)
                preds = model(flow_x, pkt_x)
                val_loss += criterion(preds, y).item()
                val_correct += (preds.argmax(1) == y).sum().item()
        val_acc = val_correct / len(val_dataset)
        avg_val_loss = val_loss / len(val_loader)
        
        # 早停判断
        if avg_val_loss < best_avg_val_loss:
            best_avg_val_loss = avg_val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'DM-HNN/DM-HNN_best_model.pth')
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
    model.load_state_dict(torch.load('DM-HNN/DM-HNN_best_model.pth'))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for flow_x, pkt_x, y in test_loader:
            flow_x, pkt_x = flow_x.to(device), pkt_x.to(device)
            preds = model(flow_x, pkt_x).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    
    # 将数字标签转换回原始类别名
    true_labels = [full_raw.reverse_label_map[label] for label in all_labels]
    pred_labels = [full_raw.reverse_label_map[pred] for pred in all_preds]
    
    print("\nTest Classification Report:", flush=True)
    print(classification_report(true_labels, pred_labels, target_names=categories, digits=4), flush=True)
    
    # 混淆矩阵输出
    # 首先将标签名转换为数字索引
    test_labels_idx = [categories.index(label) for label in true_labels]
    test_preds_idx = [categories.index(label) for label in pred_labels]
    
    conf_matrix = compute_confusion_matrix(test_labels_idx, test_preds_idx, num_classes=len(categories))

    # 计算归一化混淆矩阵（按行）
    normalized_conf = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # 输出带百分比的矩阵
    print("\nNormalized Confusion Matrix (row-wise):", flush=True)
    norm_df = pd.DataFrame(
        normalized_conf,
        index=categories,
        columns=categories
    ).applymap(lambda x: f"{x:.1%}")  # 转换为百分比格式

    print(norm_df, flush=True)
    norm_df.to_csv('DM-HNN/DM-HNN_confusion_matrix.csv', index_label='True\Predicted')