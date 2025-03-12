import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import random
from byol_pytorch import BYOL

# 自定义一维增强
class AddGaussianNoise(nn.Module):
    def __init__(self, p=0.5, std=0.1):
        super().__init__()
        self.p = p
        self.std = std
    def forward(self, x):
        return x + torch.randn_like(x) * self.std if random.random() < self.p else x

class RandomMask(nn.Module):
    def __init__(self, p=0.3, mask_ratio=0.2):
        super().__init__()
        self.p = p
        self.mask_ratio = mask_ratio
    def forward(self, x):
        if random.random() < self.p:
            mask = torch.rand_like(x) > self.mask_ratio
            return x * mask
        return x

# 一维骨干网络
class MLPBackbone(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=512, output_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 最后一层是输出层
        )
    
    def forward(self, x):
        return self.layers(x)  # 直接返回Sequential的输出

# BYOL初始化（调整后）
learner = BYOL(
    net=MLPBackbone(input_dim=100),
    input_dim=100,  # 实际是输入维度
    augment_fn=T.Compose([AddGaussianNoise(), RandomMask()]),
    augment_fn2=T.Compose([AddGaussianNoise(), RandomMask()]),
    projection_size=256,
    projection_hidden_size=512
)

# 测试
x = torch.randn(32, 100)  # [batch_size=32, features=100]
loss = learner(x)
print(loss)  # 应输出损失值
print("success")