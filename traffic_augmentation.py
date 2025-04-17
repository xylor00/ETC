import torch
import torch.distributions as dist
from torch import nn

class TrafficAugmentation(nn.Module):
    """优化后的GPU加速增强模块"""
    def __init__(self, max_rtt=0.01, mss=1448, max_length=100):
        super().__init__()
        self.MAX_RTT = max_rtt
        self.MSS = mss
        self.max_length = max_length
        
    def forward(self, x):
        # 输入x形状: [batch_size, seq_len]
        # 添加通道维度 [batch_size, seq_len, 1]
        x = x.unsqueeze(-1)
        
        # 生成随机RTT和延迟（向量化操作）
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # 生成RTT张量 [batch_size, 1]
        RTTs = torch.rand(batch_size, 1, device=device) * self.MAX_RTT
        
        # 生成延迟张量 [batch_size, seq_len]
        delays = self._get_delay(batch_size, seq_len, device)
        
        # 计算累积延迟 [batch_size, seq_len]
        cum_delays = torch.cumsum(delays, dim=1)
        
        # 生成拆分掩码 [batch_size, seq_len]
        split_mask = cum_delays <= RTTs
        
        # 计算有效包长度（减去40字节头）
        pkt_lengths = x.squeeze(-1) - 40
        pkt_lengths = torch.clamp_min(pkt_lengths, 0)  # 确保非负
        
        # 计算拆分后的包数量 [batch_size, seq_len]
        split_counts = (pkt_lengths / self.MSS).ceil().int()
        
        # 生成增强后的序列（向量化填充）
        aug_flow = self._vectorized_augmentation(pkt_lengths, split_mask, split_counts)
        
        # 调整形状为 [batch_size, max_length, 1]
        return aug_flow.unsqueeze(-1)

    def _get_delay(self, batch_size, seq_len, device):
        """向量化生成延迟"""
        # 生成延迟分布（90%小延迟，10%固定大延迟）
        mask = torch.rand(batch_size, seq_len, device=device) < 0.1
        small_delays = dist.Exponential(rate=1/0.00094557476340694).sample((batch_size, seq_len)).to(device)
        fixed_delays = torch.full((batch_size, seq_len), 0.21, device=device)
        delays = torch.where(mask, fixed_delays, small_delays)
        return delays

    def _vectorized_augmentation(self, pkt_lengths, split_mask, split_counts):
        """向量化处理拆包逻辑"""
        batch_size, seq_len = pkt_lengths.shape
        device = pkt_lengths.device
        
        # 计算每个包拆分后的总长度
        split_lengths = split_counts * self.MSS
        
        # 应用RTT拆分条件
        final_lengths = torch.where(
            split_mask,
            split_lengths,
            pkt_lengths
        )
        
        # 限制最大长度并填充
        final_lengths = torch.clamp_max(final_lengths + 40, self.MSS + 40)  # 恢复40字节头
        padded = torch.zeros(batch_size, self.max_length, device=device)
        seq_len = min(seq_len, self.max_length)
        padded[:, :seq_len] = final_lengths[:, :seq_len]
        return padded

class IdentityAugmentation(nn.Module):
    def forward(self, x):
        return x.unsqueeze(-1)  # 保持形状一致