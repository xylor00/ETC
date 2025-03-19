import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import random
from byol_pytorch import BYOL
from traffic_augmentation import TrafficAugmentation, IdentityAugmentation
import pandas as pd

# 模拟拆包粘包增强
aug_1 = TrafficAugmentation()
aug_2 = IdentityAugmentation()

# 一维骨干网络
class GRUBackbone(nn.Module):
    def __init__(
        self,
        input_dim=1,       # 输入特征维度（如传感器通道数）
        hidden_dim=128,    # GRU隐藏层维度
        num_layers=2,      # GRU层数
        output_dim=256,    # 最终输出维度（需与BYOL投影头匹配）
        bidirectional=False
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,      # 输入形状为 [batch, seq_len, features]
            bidirectional=bidirectional
        )
        # 全连接层将GRU输出映射到目标维度
        self.fc = nn.Linear(
            hidden_dim * (2 if bidirectional else 1),
            output_dim
        )
        
    def forward(self, x):
        """
        输入x形状: [batch_size, seq_len, input_dim]
        输出形状: [batch_size, output_dim]
        """
        # GRU前向传播
        gru_out, _ = self.gru(x)  # gru_out形状: [batch, seq_len, hidden_dim * directions]
        
        # 取最后一个时间步的输出
        last_step_out = gru_out[:, -1, :]  # [batch, hidden_dim * directions]
        
        # 全连接层映射到目标维度
        output = self.fc(last_step_out)    # [batch, output_dim]
        return output

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