import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import random
from byol_pytorch import BYOL
from traffic_augmentation import TrafficAugmentation, IdentityAugmentation
import pandas as pd
from GRUnet import GRUBackbone
from IPhead import Get_headers
from byoltrainer import BYOL_train
from ngram import create_plevel_feature


max_length = 100

all_flows_dict = Get_headers()

flow_sequences = []#存储数据包长度序列
IPheads = []#存储数据包头内容


for flow_key, flow_data in all_flows_dict.items():
    #读取每个流的类别
    label = flow_key[-1]
    
    #读取每个流的数据包长度序列
    pkt_length_sequence = flow_data['lengths']        
     
    #将长度序列截断或填充位等长，方便后续处理    
    if len(pkt_length_sequence) < max_length:
        pkt_length_sequence += [0] * (max_length - len(pkt_length_sequence))
    else:
        pkt_length_sequence = pkt_length_sequence[:max_length]
        
    flow_sequences.append(pkt_length_sequence)
    
    
    #对每个流的IP包头数据进行n-gram处理
    IPhead_bytes = flow_data['byte']
    plevel_feature = create_plevel_feature(IPhead_bytes)
    IPheads.append(plevel_feature)
    
print(flow_sequences[0])
print(IPheads[0])

"""
#使用BYOL方法训练GRU网络
BYOL_train(flow_sequences)
"""

