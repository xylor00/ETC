import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from byol_pytorch import BYOL
from torchvision import models

import copy  # 深拷贝模块
import random  # 随机数模块
from functools import wraps  # 装饰器工具

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist  # 分布式训练

from torchvision import transforms as T  # 图像变换
 
# 自定义一维数据增强示例
class OneDAugmentation1(nn.Module):
    """一维数据增强策略1"""
    def __init__(self, noise_level=0.1, drop_prob=0.2):
        super().__init__()
        self.noise_level = noise_level
        self.drop = nn.Dropout(p=drop_prob)
        
    def forward(self, x):
        # x shape: (batch, features) 或 (batch, channels, length)
        
        # 添加高斯噪声
        if self.training:
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
            
        # 随机遮蔽部分特征
        x = self.drop(x)
        return x

class OneDAugmentation2(nn.Module):
    """一维数据增强策略2"""
    def __init__(self, shift_range=5, scale_range=(0.8, 1.2)):
        super().__init__()
        self.shift_range = shift_range  # 适用于时序数据的位移
        self.scale_range = scale_range
        
    def forward(self, x):
        # x shape: (batch, channels, length)
        
        # 随机缩放
        scale = torch.empty(x.size(0)).uniform_(*self.scale_range)
        x = x * scale.view(-1, 1, 1)
        
        # 随机位移（对时序数据）
        if x.dim() == 3 and self.shift_range > 0:
            shift = torch.randint(-self.shift_range, self.shift_range, (1,)).item()
            x = torch.roll(x, shifts=shift, dims=-1)
            
        return x

# 示例主干网络（一维版ResNet）
class OneDResNet(nn.Module):
    def __init__(self, input_dim=128, feat_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, feat_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# 初始化BYOL
input_dim = 128  # 一维向量的特征维度
model = BYOL(
    net = OneDResNet(input_dim),  # 使用一维网络
    image_size = None,  # 忽略原图像尺寸参数
    augment_fn = OneDAugmentation1(),
    augment_fn2 = OneDAugmentation2(),
    projection_size = 256,
    projection_hidden_size = 512,
    hidden_layer = -1  # 直接使用网络输出层
)

# 使用示例
batch = torch.randn(32, input_dim)  # 输入形状(batch, features)
loss = model(batch)  # 计算损失
model.update_moving_average()  # 更新目标网络