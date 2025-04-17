import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    """基于注意力的特征融合模块"""
    def __init__(self, flevel_dim, plevel_dim, hidden_dim=128, align_dim=256):

        super().__init__()
        
        # 将包级别流量映射到对齐维度
        self.p_proj = nn.Linear(plevel_dim, align_dim)
        
        # 注意力权重计算网络
        self.attention = nn.Sequential(
            nn.Linear(align_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 输出两个特征的权重
            nn.Softmax(dim=1)
        )
        
    def forward(self, f_feat, p_feat):
        
        p_feat = self.p_proj(p_feat)  # [batch, align_dim]
        
        # 拼接特征计算注意力权重
        combined = torch.cat([f_feat, p_feat], dim=1)
        weights = self.attention(combined)  # [batch, 2]
        
        # 加权融合
        fused = weights[:, 0:1] * f_feat + weights[:, 1:2] * p_feat
        return fused

# 加载特征数据
f_features = pd.read_csv('dataset/flevel_features.csv', skiprows=1, header=None)
p_features = pd.read_csv('dataset/plevel_features.csv', skiprows=1, header=None)

flevel_features = f_features.iloc[:, :-1].values.astype(np.float32)
plevel_features = p_features.iloc[:, :-1].values.astype(np.float32)
labels = f_features.iloc[:, -1].values

# 转换为PyTorch张量
f_tensor = torch.from_numpy(flevel_features)
p_tensor = torch.from_numpy(plevel_features)

# 初始化融合模块
fusion_model = FeatureFusion(
    flevel_dim=f_tensor.shape[1], 
    plevel_dim=p_tensor.shape[1]
)

# 执行融合
with torch.no_grad():
    merge_features = fusion_model(f_tensor, p_tensor).numpy()

# 添加标签并保存
df_merge = pd.DataFrame(merge_features)
df_merge['label'] = labels
df_merge.to_csv('dataset/merge_features.csv', index=False)