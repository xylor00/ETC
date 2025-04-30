import torch
import random
from scipy.stats import expon
from torch import nn
from torch.distributions import Exponential
import torch.nn.functional as F

class TrafficAugmentation(nn.Module):
    """PyTorch适配的流量数据增强模块"""
    def __init__(self, max_rtt=0.01, mss=1448, max_length=100):
        super().__init__()
        self.MAX_RTT = max_rtt
        self.MSS = mss
        self.max_length = max_length
        
    def forward(self, x):
           
        # 执行流量增强
        aug_flow_torch = self._traffic_augmentation(x)       
            
        return aug_flow_torch.unsqueeze(-1)

    def _traffic_augmentation(self, x):
        # 生成延迟向量
        batch_size, seq_len = x.shape
        delays = self._get_delay(batch_size, seq_len)

        #存储处理后的数据
        final_lengthflows = torch.tensor([])
    
        #数据增强处理
        for i in range(batch_size):
            #存储处理后的长度序列
            lengthflow = torch.tensor([])
            #标记当前处理的位置
            j = 0

            while j < seq_len and x[i][j].item() > 0:# 当前流还没读取完毕，或者当前流剩余数据包长度均为0
                RTT = random.random() * self.MAX_RTT
                buf = 0

                while j < seq_len and RTT > 0:
                    RTT -= delays[i][j].item()
                    buf += x[i][j].item()# 数据预处理时以及减去数据包头长度，因此此处不用再减去40

                    j += 1
                
                while buf > self.MSS:
                    lengthflow = torch.cat((lengthflow, torch.tensor([self.MSS + 40])), 0)
                    buf -= self.MSS

                if buf > 0:
                    lengthflow = torch.cat((lengthflow, torch.tensor([buf + 40])), 0)

            # 填充/截断到固定长度
            fit_lengthflow = F.pad(lengthflow[:seq_len], (0, max(seq_len - lengthflow.size(0), 0)))
            
            final_lengthflows = torch.cat((final_lengthflows, fit_lengthflow.view(1, -1)), 0)

        return final_lengthflows
            

    def _get_delay(self, batch_size, seq_len):
        # 生成延迟张量（向量化实现）
        mask = torch.rand((batch_size, seq_len)) < 0.1
        loc = 1e-06
        scale = 0.00094557476340694

        # 指数分布采样
        exponential = Exponential(1 / scale)
        delays_exp = loc + exponential.sample((batch_size, seq_len))

        # 合并延迟
        delays = torch.where(mask, torch.full_like(delays_exp, 0.21), delays_exp)
        return delays

    def _fit_data(self, data):
        """调整序列长度"""
        if len(data) < self.max_length:
            data += [0] * (self.max_length - len(data))
        else:
            data = data[:self.max_length]
        return data

class IdentityAugmentation(nn.Module):
    """原始数据（无增强）"""
    def forward(self, x):
        return x.unsqueeze(-1)# 直接返回原始输入