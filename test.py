import torch
from byol_pytorch import BYOL
from torchvision import models

resnet = models.resnet50(pretrained=True)


learner = BYOL(
    resnet,
    image_size = 256,
    hidden_layer = 'avgpool'
)

state_dict1 = learner.state_dict()
for key, data in state_dict1.items():
    print(f"Parameter {key} =  {data}.")

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.randn(5, 3, 256, 256)

for _ in range(10):
    images = sample_unlabelled_images()
    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average() # update moving average of target encoder
 
# 获取状态字典
print("---------------------------------------After training-------------------------------------------")
state_dict2 = learner.state_dict()
 
for key, data in state_dict2.items():
    print(f"Parameter {key} =  {data}.")