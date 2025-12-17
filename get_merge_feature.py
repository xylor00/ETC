import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import OneCycleLR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureFusion(nn.Module):
    """特征融合模块"""
    def __init__(self, flevel_dim, plevel_dim, hidden_dim=128, align_dim=256, num_classes=6):
        super().__init__()
        self.f_proj = nn.Linear(flevel_dim, align_dim)
        self.p_proj = nn.Linear(plevel_dim, align_dim)
        self.fuse = nn.Sequential(
            nn.Linear(align_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Linear(align_dim, num_classes)
        
    def forward(self, f_feat, p_feat):
        f_feat = self.f_proj(f_feat)
        p_feat = self.p_proj(p_feat)
        
        #计算动态权重
        combined = torch.cat([f_feat, p_feat], dim=1)
        weights = self.fuse(combined)
        
        #加权融合
        fused = weights[:, 0:1] * f_feat + weights[:, 1:2] * p_feat
        
        #预测值，用于训练
        logits = self.classifier(fused)
        return fused, logits

f_reader = pd.read_csv('features/flevel_features.csv', skiprows=1, header=None)
p_reader = pd.read_csv('features/plevel_features.csv', skiprows=1, header=None)

# 确定特征维度（最后一列是标签 'label'）
flevel_dim = f_reader.shape[1] - 1
plevel_dim = p_reader.shape[1] - 1

# 提取特征并转换为 float32 张量
f_feat_tensor = torch.from_numpy(f_reader.iloc[:, :-1].values.astype(np.float32)).to(device)
p_feat_tensor = torch.from_numpy(p_reader.iloc[:, :-1].values.astype(np.float32)).to(device)
#读取标签
raw_labels = f_reader.iloc[:, -1].values.astype(str)
le = LabelEncoder()
integer_labels = le.fit_transform(raw_labels)
labels_tensor = torch.from_numpy(integer_labels).long().to(device)


#初始化融合模块

model = FeatureFusion(flevel_dim=flevel_dim, plevel_dim=plevel_dim,num_classes=len(le.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
# 优化器为AdamW
lr = 1e-4
num_epochs = 5000
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,                  # 设置基础学习率
    weight_decay=1e-3,
    betas=(0.9, 0.98)
)

scheduler = OneCycleLR(
    optimizer,
    max_lr=lr*20,      # 峰值学习率
    total_steps=num_epochs,
    pct_start=0.3,     # warmup阶段
    anneal_strategy='cos',
    div_factor=10,     # 初始学习率与峰值比率
    final_div_factor=1e4
)

# 早停参数
best_avg_val_loss = 100
patience = 100
no_improve_epochs = 0
stop_training = False

print("Training begins to optimize the fusion weights...")

for epoch in range(num_epochs):  
    # 检查早停条件
    if stop_training:
        print(f"Early stopping at epoch {epoch}", flush=True)
        print(f"get model at epoch {epoch-patience}", flush=True)
        break
    
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    fused_feat, logits = model(f_feat_tensor, p_feat_tensor)
    
    # 计算损失：根据标签引导模型学习哪些特征更重要
    loss = criterion(logits, labels_tensor)
    
    # 反向传播与优化
    loss.backward()
    optimizer.step()
    
    if loss.item() < best_avg_val_loss:
        best_avg_val_loss = loss.item()
        no_improve_epochs = 0
    
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            stop_training = True
            
    scheduler.step()  # 执行主调度
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Best Loss: {best_avg_val_loss:.4f}", flush=True)
        
model.eval() # 切换到评估模式
with torch.no_grad():
    # 获取最终优化后的融合特征和权重比例
    final_fused, _ = model(f_feat_tensor, p_feat_tensor)
    
    # 移回 CPU 并转为 NumPy
    merge_features = final_fused.cpu().numpy()

# 构建结果数据框
df_merge = pd.DataFrame(merge_features)
df_merge['label'] = raw_labels

# 保存文件
df_merge.to_csv('features/merge_features.csv', index=False)