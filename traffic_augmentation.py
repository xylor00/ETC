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
            
        # 填充/截断到固定长度
        final_flow = self._fit_data(aug_flow)
            
        return final_flow

    def _traffic_augmentation(self, O):
        """核心增强逻辑"""
        lengthflow = []
        interval = 0
        i = 0
        num = len(O) - 1
        delays = self._get_delay(len(O))
        
        while num >= i and O[i] > 0:
            RTT = random.random() * self.MAX_RTT
            buf = 0
            while num >= i and RTT > 0:
                interval = delays[i]
                RTT -= interval
                buf += O[i] - 40  # 假设包含40字节头
                i += 1
                
            while buf > self.MSS:
                lengthflow.append(self.MSS + 40)
                buf -= self.MSS
                
            if buf > 0:
                lengthflow.append(buf + 40)
                
        return lengthflow

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

    def _fit_data(self, seq):
        """调整序列长度"""
        data = seq[:-1]
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
        return x  # 直接返回原始输入