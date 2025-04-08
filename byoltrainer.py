import torch
from byol_pytorch import BYOL
from traffic_augmentation import TrafficAugmentation, IdentityAugmentation
import pandas as pd
from GRUnet import GRUBackbone


def BYOL_train(flevel_features):

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    for i in range(5):
        flow = flevel_features[i : i + 2]
        f_tensor = torch.from_numpy(flow)
        loss = learner(f_tensor)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder
        print(loss)  # 应输出损失值


    # 测试
    torch.save(gru_backbone.state_dict(), './improved-net.pt')

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
    input_dim=100,
    augment_fn=aug_1,
    augment_fn2=aug_2,
    projection_size=256,
    projection_hidden_size=512
)

flow_sequences = pd.read_csv('dataset/flow_sequences.csv', skiprows=1, header=None)

flevel_features = flow_sequences.iloc[:, :-1].values.astype(int)
labels = flow_sequences.iloc[:, -1].values

BYOL_train(flevel_features)