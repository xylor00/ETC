import torch
import torch.nn as nn
from torch.distributions import Exponential

class TrafficAugmentation(nn.Module):
    """向量化实现的流量数据增强模块"""
    def __init__(self, max_rtt=0.01, mss=1448, max_length=100):
        super().__init__()
        self.MAX_RTT = max_rtt
        self.MSS = mss
        self.max_length = max_length
        self.expon_scale = 0.00094557476340694
        self.expon_loc = 1e-06

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        device = x.device

        # 生成延迟矩阵
        delays = self._generate_delays(batch_size, seq_len, device)

        # 生成随机RTT矩阵
        rtts = torch.rand((batch_size, 1), device=device) * self.MAX_RTT

        # 向量化处理
        processed = self._vectorized_process(x, delays, rtts, seq_len, device)

        # 调整到固定长度
        padded = self._pad_sequences(processed)
        
        return padded.unsqueeze(-1)

    def _generate_delays(self, batch_size, seq_len, device):
        # 生成延迟张量
        probs = torch.rand((batch_size, seq_len), device=device)
        delays = torch.where(
            probs < 0.1,
            torch.full((batch_size, seq_len), 0.21, device=device),
            self.expon_loc + Exponential(1/self.expon_scale).sample((batch_size, seq_len)).to(device)
        )
        return delays

    def _vectorized_process(self, x, delays, rtts, seq_len, device):
        batch_size = x.shape[0]
        all_packets = []

        cum_delays = torch.zeros_like(delays)
        cum_delays[:, 1:] = torch.cumsum(delays[:, :-1], dim=1)
        
        valid_mask = x > 0
        
        for _ in range(seq_len):
            window_mask = (cum_delays < rtts) & valid_mask
            bufs = (x * window_mask).sum(dim=1).long()  # 确保整型
            
            full_packets = (bufs // self.MSS).clamp(min=0)
            remainders = bufs % self.MSS
            
            packets = []
            for i in range(batch_size):
                full_count = int(full_packets[i].item())  # 双重转换确保整型
                remainder = int(remainders[i].item())
                
                pkts = [self.MSS + 40] * full_count
                if remainder > 0:
                    pkts.append(remainder + 40)
                packets.append(torch.tensor(pkts, device=device, dtype=x.dtype))
            
            # 更新状态
            max_packets = max(len(p) for p in packets) if packets else 0
            if max_packets == 0:
                break
                
            # 更新累积延迟
            cum_delays = cum_delays + delays * (~window_mask).float()
            valid_mask = valid_mask & (~window_mask)
            
            # 生成新的RTT
            rtts = torch.rand((batch_size, 1), device=device) * self.MAX_RTT
            
            # 收集结果
            batch_packets = torch.zeros((batch_size, max_packets), device=device)
            for i, p in enumerate(packets):
                if len(p) > 0:
                    batch_packets[i, :len(p)] = p
            all_packets.append(batch_packets)
            
            if not valid_mask.any():
                break

        # 合并所有数据包
        if all_packets:
            concatenated = torch.cat(all_packets, dim=1)
            return concatenated[:, :self.max_length]
        return torch.zeros((batch_size, self.max_length), device=device)

    def _pad_sequences(self, sequences):
        # 自动填充/截断到固定长度
        if sequences.shape[1] < self.max_length:
            pad = self.max_length - sequences.shape[1]
            return torch.nn.functional.pad(sequences, (0, pad))
        return sequences[:, :self.max_length]

class IdentityAugmentation(nn.Module):
    """原始数据（无增强）"""
    def forward(self, x):
        return x.unsqueeze(-1)
    



input = [[46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[46,46,46,46,40,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[40,40,40,40,255,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]


input_torch = torch.Tensor(input)

aug = TrafficAugmentation()

output = aug(input_torch)


print(input_torch)
print(output)
