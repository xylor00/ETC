import os
import pandas as pd
from traffic_augmentation import TrafficAugmentation, IdentityAugmentation
import numpy as np
import torch
from byol_pytorch import BYOL
from torchvision import models

def main():
    dataset = pd.read_csv("test.csv", skiprows=1, header=None)
    
    # 分离特征和标签
    features = dataset.iloc[:, :-1].values.astype(int)
    labels = dataset.iloc[:, -1].values
    f_tensor = torch.from_numpy(features)

    O = TrafficAugmentation()
    a = O(f_tensor)
    print(a)

if __name__ == '__main__':
    main()