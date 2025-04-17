import torch.nn as nn

class GRUBackbone(nn.Module):
    def __init__(
        self,
        input_dim=1,       # 输入特征维度（如传感器通道数）
        hidden_dim=128,    # GRU隐藏层维度
        num_layers=2,      # GRU层数
        output_dim=256,    # 最终输出维度（需与BYOL投影头匹配）
        bidirectional=False, # 是否双向GRU
        dropout=0.2  # dropout参数
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,# 输入形状为 [batch, seq_len, features]
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0  # 多层GRU时启用dropout
        )
        # 新增LayerNorm和Dropout
        self.post_gru_norm = nn.LayerNorm(hidden_dim * (2 if bidirectional else 1))
        self.post_gru_dropout = nn.Dropout(p=dropout)
        
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
        gru_out, _ = self.gru(x)# gru_out形状: [batch, seq_len, hidden_dim * directions]
        
        # 取最后一个时间步的输出
        last_step_out = gru_out[:, -1, :]# [batch, hidden_dim * directions]
        
        # 正则化层
        normalized = self.post_gru_norm(last_step_out)
        normalized = self.post_gru_dropout(normalized)
        
        # 全连接层映射到目标维度
        output = self.fc(normalized)# [batch, output_dim]
        return output