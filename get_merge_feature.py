import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    """基于注意力的特征融合模块"""
    def __init__(self, flevel_dim, plevel_dim, hidden_dim=128, align_dim=256):
        super().__init__()
        self.p_proj = nn.Linear(plevel_dim, align_dim)
        self.attention = nn.Sequential(
            nn.Linear(align_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, f_feat, p_feat):
        p_feat = self.p_proj(p_feat)
        combined = torch.cat([f_feat, p_feat], dim=1)
        weights = self.attention(combined)
        fused = weights[:, 0:1] * f_feat + weights[:, 1:2] * p_feat
        return fused

# 确定特征维度
f_first_row = pd.read_csv('features/flevel_features.csv', skiprows=1, nrows=1, header=None)
flevel_dim = f_first_row.shape[1] - 1  # 最后一列是标签

p_first_row = pd.read_csv('features/plevel_features.csv', skiprows=1, nrows=1, header=None)
plevel_dim = p_first_row.shape[1] - 1  # 原代码中去掉了最后一列

# 初始化融合模块
fusion_model = FeatureFusion(flevel_dim=flevel_dim, plevel_dim=plevel_dim)

# 分块处理
chunk_size = 1000  # 根据内存调整块大小
f_reader = pd.read_csv('features/flevel_features.csv', skiprows=1, header=None, chunksize=chunk_size)
p_reader = pd.read_csv('features/plevel_features.csv', skiprows=1, header=None, chunksize=chunk_size)

with torch.no_grad():
    first_chunk = True
    for f_chunk, p_chunk in zip(f_reader, p_reader):
        # 检查列数是否正确
        assert f_chunk.shape[1] == flevel_dim + 1, "Flevel chunk列数不匹配"
        assert p_chunk.shape[1] == plevel_dim + 1, "Plevel chunk列数不匹配"
        
        # 提取特征和标签
        flevel_feat = f_chunk.iloc[:, :-1].values.astype(np.float32)
        labels_block = f_chunk.iloc[:, -1].values
        plevel_feat = p_chunk.iloc[:, :-1].values.astype(np.float32)
        
        # 转换为张量并进行融合
        f_tensor = torch.from_numpy(flevel_feat)
        p_tensor = torch.from_numpy(plevel_feat)
        merge_features_block = fusion_model(f_tensor, p_tensor).numpy()
        
        # 构建并保存结果块
        df_merge_block = pd.DataFrame(merge_features_block)
        df_merge_block['label'] = labels_block
        
        # 写入文件（首次包含表头，后续追加）
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        df_merge_block.to_csv('features/merge_features.csv', index=False, mode=mode, header=header)
        first_chunk = False