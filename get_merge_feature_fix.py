import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import OneCycleLR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionFeatureFusion(nn.Module):
    def __init__(self, flevel_dim, plevel_dim, align_dim=256, num_heads=8, num_classes=6):
        super().__init__()
        # 1. 维度对齐层 (Projector)
        self.f_proj = nn.Linear(flevel_dim, align_dim)
        self.p_proj = nn.Linear(plevel_dim, align_dim)
        
        # 2. 多头注意力层 (Cross-Attention)
        # 我们使用 f_feat 去 Query p_feat 的信息
        self.attn = nn.MultiheadAttention(embed_dim=align_dim, num_heads=num_heads, batch_first=True)
        
        # 3. 层归一化与前馈网络 (类似 Transformer Block)
        self.norm = nn.LayerNorm(align_dim)
        self.ffn = nn.Sequential(
            nn.Linear(align_dim, align_dim * 2),
            nn.ReLU(),
            nn.Linear(align_dim * 2, align_dim)
        )
        
        # 4. 分类器
        self.classifier = nn.Linear(align_dim, num_classes)
        
    def forward(self, f_feat, p_feat):
        # 线性投影
        f_feat = self.f_proj(f_feat) # [batch, align_dim]
        p_feat = self.p_proj(p_feat) # [batch, align_dim]
        
        # 为符合 MultiheadAttention 的输入 (batch_first=True): [batch, seq_len, dim]
        # 这里我们将特征视为长度为 1 的序列
        q = f_feat.unsqueeze(1) 
        k = v = p_feat.unsqueeze(1)
        
        # 交叉注意力计算
        # attn_output: 基于 p 增强后的 f 特征
        attn_output, attn_weights = self.attn(q, k, v)
        
        # 残差连接与归一化
        fused = self.norm(attn_output.squeeze(1) + f_feat)
        
        # 前馈网络进一步增强
        fused = self.ffn(fused) + fused
        
        # 预测
        logits = self.classifier(fused)
        
        return fused, logits

f_reader = pd.read_csv('features/flevel_features.csv', skiprows=1, header=None)
p_reader = pd.read_csv('features/plevel_features.csv', skiprows=1, header=None)

# 确定特征维度（最后一列是标签 'label'）
flevel_dim = f_reader.shape[1] - 1
plevel_dim = p_reader.shape[1] - 1

# 提取特征并转换为 float32 张量
f_feat_tensor = torch.from_numpy(f_reader.iloc[:, :-1].values.astype(np.float32)).to(device)
p_feat_tensor = torch.from_numpy(p_reader.iloc[:, :-1].values.astype(np.float32)).to(device)
#读取标签
raw_labels = f_reader.iloc[:, -1].values.astype(str)
le = LabelEncoder()
integer_labels = le.fit_transform(raw_labels)
labels_tensor = torch.from_numpy(integer_labels).long().to(device)


#初始化融合模块

model = AttentionFeatureFusion(flevel_dim=flevel_dim, plevel_dim=plevel_dim,num_classes=len(le.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
# 优化器为AdamW
lr = 1e-4
num_epochs = 10000
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,                  # 设置基础学习率
    weight_decay=1e-3,
    betas=(0.9, 0.98)
)

scheduler = OneCycleLR(
    optimizer,
    max_lr=lr*20,      # 峰值学习率
    total_steps=num_epochs,
    pct_start=0.3,     # warmup阶段
    anneal_strategy='cos',
    div_factor=10,     # 初始学习率与峰值比率
    final_div_factor=1e4
)

# 早停参数
best_avg_val_loss = 100
patience = 100
no_improve_epochs = 0
stop_training = False

print("Training begins to optimize the fusion weights...")

for epoch in range(num_epochs):  
    # 检查早停条件
    if stop_training:
        print(f"Early stopping at epoch {epoch}", flush=True)
        print(f"get model at epoch {epoch-patience}", flush=True)
        break
    
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    fused_feat, logits = model(f_feat_tensor, p_feat_tensor)
    
    # 计算损失：根据标签引导模型学习哪些特征更重要
    loss = criterion(logits, labels_tensor)
    
    # 反向传播与优化
    loss.backward()
    optimizer.step()
    
    if loss.item() < best_avg_val_loss:
        best_avg_val_loss = loss.item()
        no_improve_epochs = 0
        torch.save(model.state_dict(), 'model/merge_model.pth')
    
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            stop_training = True
            
    scheduler.step()  # 执行主调度
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Best Loss: {best_avg_val_loss:.4f}", flush=True)

model.load_state_dict(torch.load('model/merge_model.pth'))        
model.eval() # 切换到评估模式
with torch.no_grad():
    # 获取最终优化后的融合特征和权重比例
    final_fused, _ = model(f_feat_tensor, p_feat_tensor)
    
    # 移回 CPU 并转为 NumPy
    merge_features = final_fused.cpu().numpy()

# 构建结果数据框
df_merge = pd.DataFrame(merge_features)
df_merge['label'] = raw_labels

# 保存文件
df_merge.to_csv('features/merge_features.csv', index=False)