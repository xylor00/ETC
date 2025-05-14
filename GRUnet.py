import torch.nn as nn

class GRUBackbone(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        num_layers=2,
        output_dim=256,
        bidirectional=False,
        dropout=0.2,
        pooling_type='last'  # 新增池化方式参数
    ):
        super().__init__()
        self.directions = 2 if bidirectional else 1
        self.pooling_type = pooling_type

        # 输入归一化层
        self.input_norm = nn.LayerNorm(input_dim)
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 后GRU处理
        self.post_gru_norm = nn.LayerNorm(hidden_dim * self.directions)
        self.post_gru_dropout = nn.Dropout(p=dropout)
        
        # 增强的多层投影头
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * self.directions, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        """参数初始化"""
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)  # GRU权重正交初始化
        for layer in [self.fc[0], self.fc[-1]]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        # 输入形状检查
        assert len(x.shape) == 3, f"Input shape must be [batch, seq_len, input_dim], got {x.shape}"
        
        # 输入归一化
        x = self.input_norm(x)
        
        # GRU前向
        gru_out, _ = self.gru(x)  # [batch, seq_len, hidden * directions]
        
        # 池化策略
        if self.pooling_type == 'last':
            pooled = gru_out[:, -1, :]
        elif self.pooling_type == 'mean':
            pooled = gru_out.mean(dim=1)
        elif self.pooling_type == 'max':
            pooled = gru_out.max(dim=1)[0]
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
        
        # 后处理
        normalized = self.post_gru_norm(pooled)
        normalized = self.post_gru_dropout(normalized)
        
        # 投影头
        output = self.fc(normalized)
        return output