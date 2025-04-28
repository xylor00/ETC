import torch
import random
from scipy.stats import expon
from torch import nn

class TrafficAugmentation(nn.Module):
    """PyTorch适配的流量数据增强模块"""
    def __init__(self, max_rtt=0.01, mss=1448, max_length=100):
        super().__init__()
        self.MAX_RTT = max_rtt
        self.MSS = mss
        self.max_length = max_length
        
    def forward(self, x):
           
        # 执行流量增强
        aug_flow = self._traffic_augmentation(x)
        aug_flow_torch = torch.Tensor(aug_flow)
        final_flow = torch.reshape(aug_flow_torch, shape=(aug_flow_torch.shape[0], aug_flow_torch.shape[1], 1))
            
        return final_flow

    def _traffic_augmentation(self, x):
        
        O = x.numpy()
        final_lengthflows = []
        """核心增强逻辑"""
        for row in O:# 对单独流的处理
            lengthflow = []
            fit_lengthflow = []
            interval = 0
            i = 0
            num = len(row) - 1
            delays = self._get_delay(len(row))
            
            while num >= i and row[i] > 0:# 当前流还没读取完毕，或者当前流剩余数据包长度均为0
                RTT = random.random() * self.MAX_RTT
                buf = 0
                while num >= i and RTT > 0:
                    interval = delays[i]
                    RTT -= interval
                    buf += row[i]# 数据预处理时以及减去数据包头长度，因此此处不用再减去40
                    i += 1
                    
                while buf > self.MSS:
                    lengthflow.append(self.MSS + 40)
                    buf -= self.MSS
                    
                if buf > 0:
                    lengthflow.append(buf + 40)
                   
            # 填充/截断到固定长度
            fit_lengthflow = self._fit_data(lengthflow)             

            final_lengthflows.append(fit_lengthflow)
        
        return final_lengthflows

    def _get_delay(self, size):
        """生成延迟序列"""
        delays = []
        while len(delays) < size:
            if random.random() < 0.1:
                delays.append(0.21)
            else:
                loc, scale = 1e-06, 0.00094557476340694
                delays.extend(expon.rvs(loc=loc, scale=scale, size=1))
        return delays[:size]

    def _fit_data(self, data):
        """调整序列长度"""
        if len(data) < self.max_length:
            data += [0] * (self.max_length - len(data))
        else:
            data = data[:self.max_length]
        return data

class IdentityAugmentation(nn.Module):
    """原始数据（无增强）"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        x = torch.reshape(x, shape=(x.shape[0], x.shape[1], 1))
        return x  # 直接返回原始输入
    


input = [[46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[46,46,46,46,40,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[40,40,40,40,255,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]


input_torch = torch.Tensor(input)

aug = TrafficAugmentation()

output = aug(input_torch)


print(input_torch)
print(output)