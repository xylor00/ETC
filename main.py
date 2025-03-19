import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import random
from byol_pytorch import BYOL
from traffic_augmentation import TrafficAugmentation, IdentityAugmentation
import pandas as pd
from GRUnet import GRUBackbone

# 模拟拆包粘包增强
aug_1 = TrafficAugmentation()
aug_2 = IdentityAugmentation()

# 初始化GRU骨干网络
gru_backbone = GRUBackbone(
    input_dim=1,          # 假设输入是单变量时间序列
    hidden_dim=128,
    output_dim=256        # 需与projection_size一致
)

# 创建BYOL实例
learner = BYOL(
    net=gru_backbone,
    input_dim=100,        # 此处参数名需重命名（如input_size），但为兼容保留
    augment_fn=aug_1,
    augment_fn2=aug_2,
    projection_size=256,
    projection_hidden_size=512
)

dataset = pd.read_csv("test.csv", skiprows=1, header=None)

# 分离特征和标签
features = dataset.iloc[:, :-1].values.astype(int)
labels = dataset.iloc[:, -1].values

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

for i in range(5):
    flow = features[i : i + 2]
    f_tensor = torch.from_numpy(flow)
    loss = learner(f_tensor)
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average() # update moving average of target encoder
    print(loss)  # 应输出损失值


# 测试
torch.save(gru_backbone.state_dict(), './improved-net.pt')