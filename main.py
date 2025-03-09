import os
import pandas as pd
from traffic_augmentation import TrafficAugmentation, IdentityAugmentation
from byol_pytorch import BYOL

import torch
from torch import nn

# 假设原始数据集由3个一维向量组成，每个序列长度100
O = torch.randint(0, 1000, (3, 100)).float()  # (3个样本, 100维)

# 创建增强器实例 --------------------------------------------------
max_length = 128  # 与TrafficAugmentation参数保持一致
traffic_aug = TrafficAugmentation(max_length=max_length)
identity_aug = IdentityAugmentation()

# 定义网络结构 ----------------------------------------------------
class TrafficEncoder(nn.Module):
    """处理一维流量特征的编码器"""
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # 输出特征维度
        )
    
    def forward(self, x):
        return self.net(x)

# 初始化模型 ------------------------------------------------------
encoder = TrafficEncoder(input_dim=max_length)
projection_size = 128
projection_hidden = 512

# 构建BYOL对比学习框架 --------------------------------------------
byol_model = BYOL(
    net = encoder,
    image_size = 32,       # 此参数不再实际使用
    hidden_layer = -1,     # 直接使用网络最终输出
    projection_size = projection_size,
    projection_hidden_size = projection_hidden,
    augment_fn = traffic_aug,    # 主要增强
    augment_fn2 = identity_aug,  # 辅助增强（原始数据）
    use_momentum = True
)

# 数据预处理检查 --------------------------------------------------
def validate_data_shape(x):
    """确保输入符合(batch, seq_len)格式"""
    assert len(x.shape) == 2, "输入必须是二维张量 (batch, sequence)"
    return x

# 使用示例 --------------------------------------------------------
batch = validate_data_shape(O)  # (3, 100)

# 前向计算
loss = byol_model(batch)
print(f"对比损失值: {loss.item():.4f}")

# 反向传播
loss.backward()

# 更新动量编码器
byol_model.update_moving_average()