import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import multiprocessing
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
categories = ["socialapp", "chat", "email", "file", "streaming", "VoIP"]

def compute_confusion_matrix(true_labels, pred_labels, num_classes):
    """计算混淆矩阵"""
    true_labels = torch.as_tensor(true_labels, dtype=torch.long)
    pred_labels = torch.as_tensor(pred_labels, dtype=torch.long)
    
    assert torch.all(true_labels >= 0) & torch.all(true_labels < num_classes)
    assert torch.all(pred_labels >= 0) & torch.all(pred_labels < num_classes)
    
    indices = true_labels * num_classes + pred_labels
    matrix = torch.bincount(
        indices,
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    
    return matrix.numpy()

class RawDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path, skiprows=1, header=None)
        self.features = self.data.iloc[:, :-1].values.astype('float32')
        self.labels = self.data.iloc[:, -1].values

    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx): 
        return self.features[idx], self.labels[idx]

class StandardizedDataset(Dataset):
    def __init__(self, raw_dataset, indices, mean, std):
        self.features = (raw_dataset.features[indices] - mean) / (std + 1e-8)
        self.labels = [categories.index(raw_dataset.labels[i]) for i in indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx]),
            torch.tensor(self.labels[idx])
        )

class MultiBiGRU(nn.Module):
    """多层双向GRU"""
    def __init__(self, input_size, hidden_size, num_layers, dropout, is_cat=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.is_cat = is_cat
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 初始化隐藏状态参数
        self.init_h = nn.Parameter(torch.zeros(2 * num_layers, 1, hidden_size))
        
    def forward(self, x, seq_lens, init_h=None):
        batch_size = x.size(0)
        if init_h is None:
            init_h = self.init_h.repeat(1, batch_size, 1)
        
        # 打包序列处理变长
        packed = pack_padded_sequence(x, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        
        # 通过GRU
        outputs, hidden = self.gru(packed, init_h)
        
        # 解包序列
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        
        # 处理隐藏状态
        hidden = hidden.permute(1, 0, 2).contiguous().view(batch_size, -1)
        
        return hidden, outputs

class FSNet(nn.Module):
    """流序列网络"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 编码器GRU
        self.encoder = MultiBiGRU(
            input_size=config.input_size,
            hidden_size=config.hidden,
            num_layers=config.layer,
            dropout=1 - config.keep_prob,
            is_cat=True
        )
        
        # 解码器GRU
        self.decoder = MultiBiGRU(
            input_size=2 * config.hidden * config.layer,
            hidden_size=config.hidden,
            num_layers=config.layer,
            dropout=1 - config.keep_prob,
            is_cat=True
        )
        
        # 特征压缩层
        self.compress = nn.Sequential(
            nn.Linear(4 * config.hidden * config.layer, 2 * config.hidden),
            nn.SELU(),
            nn.Dropout(1 - config.keep_prob) if config.keep_prob < 1 else nn.Identity()
        )
        
        # 重构层
        self.reconstruct = nn.Sequential(
            nn.Linear(2 * config.hidden, config.hidden),
            nn.SELU(),
            nn.Linear(config.hidden, config.input_size)
        )
        
        # 分类层
        self.classifier = nn.Linear(2 * config.hidden, config.class_num)
    
    def forward(self, x, labels=None):
        # 创建序列长度数组（所有序列长度=1）
        batch_size = x.size(0)
        seq_lens = torch.ones(batch_size, dtype=torch.long, device=x.device)
        
        # 重塑输入为序列格式
        x = x.unsqueeze(1)
        
        # 编码器
        e_hidden, e_outputs = self.encoder(x, seq_lens)
        
        # 解码器输入
        decoder_input = e_hidden.unsqueeze(1)
        
        # 解码器
        d_hidden, d_outputs = self.decoder(decoder_input, seq_lens)
        
        # 特征拼接和压缩
        features = torch.cat([e_hidden, d_hidden], dim=1)
        compressed = self.compress(features)
        
        # 重构损失
        rec_output = self.reconstruct(d_outputs)
        rec_loss = F.mse_loss(rec_output, x)
        
        # 分类
        class_logits = self.classifier(compressed)
        preds = torch.argmax(class_logits, dim=1)
        
        # 分类损失
        class_loss = F.cross_entropy(class_logits, labels) if labels is not None else None
        
        # 总损失
        total_loss = class_loss + self.config.rec_loss * rec_loss if labels is not None else None
        
        return {
            "logits": class_logits,
            "preds": preds,
            "loss": total_loss,
            "class_loss": class_loss,
            "rec_loss": rec_loss
        }

# 配置类
class FSNetConfig:
    def __init__(self):
        self.input_size = 100  # 输入特征维度
        self.hidden = 128      # GRU隐藏单元数
        self.layer = 2         # GRU层数
        self.keep_prob = 0.7   # dropout保留率
        self.class_num = len(categories)  # 分类类别数
        self.rec_loss = 1    # 重构损失权重

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # 超参数
    input_size = 100  
    num_classes = len(categories)
    num_epochs = 500
    batch_size = 512
    lr = 0.0005

    # 加载原始数据
    full_raw = RawDataset('features/flow_sequences.csv')
    
    train_size = int(0.6 * len(full_raw))
    val_size = int(0.2 * len(full_raw))
    test_size = len(full_raw) - train_size - val_size

    # 分割数据集
    train_subset, val_subset, test_subset = random_split(
        full_raw, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(37)
    )
    
    # 获取各子集的索引
    train_indices = train_subset.indices
    val_indices = val_subset.indices
    test_indices = test_subset.indices

    # 计算训练集的均值和标准差
    train_features = full_raw.features[train_indices]
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)

    # 创建标准化数据集
    train_dataset = StandardizedDataset(full_raw, train_indices, mean, std)
    val_dataset = StandardizedDataset(full_raw, val_indices, mean, std)
    test_dataset = StandardizedDataset(full_raw, test_indices, mean, std)
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)

    # 初始化FSNet模型
    config = FSNetConfig()
    model = FSNet(config).to(device)
    
    # 损失函数和优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    # 早停参数
    best_avg_val_loss = 100
    patience = 15
    no_improve_epochs = 0
    stop_training = False
    
    for epoch in range(num_epochs):
        if stop_training:
            print(f"Early stopping at epoch {epoch}", flush=True)
            print(f"get model at epoch {epoch-patience}", flush=True)
            break
        
        # 训练阶段
        model.train()
        train_loss = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # FSNet前向传播
            outputs = model(x, y)
            loss = outputs["loss"]
            
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            
        avg_train_loss = np.mean(train_loss)
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x, y)
                val_loss += outputs["loss"].item()
                val_correct += (outputs["preds"] == y).sum().item()
                
        val_acc = val_correct / len(val_dataset)
        avg_val_loss = val_loss / len(val_loader)
        
        # 早停判断
        if avg_val_loss < best_avg_val_loss:
            best_avg_val_loss = avg_val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'FSNet/fsnet_best_model.pth')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                stop_training = True
        
        # 打印信息
        print(f"Epoch {epoch+1}: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, "
              f"Val Acc={val_acc:.2%}", flush=True)
        
    # 测试评估
    model.load_state_dict(torch.load('FSNet/fsnet_best_model.pth'))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            preds = outputs["preds"].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    
    print("\nTest Classification Report:", flush=True)
    print(classification_report(all_labels, all_preds, target_names=categories, digits=4), flush=True)
    
    # 混淆矩阵输出
    conf_matrix = compute_confusion_matrix(all_labels, all_preds, num_classes=len(categories))
    normalized_conf = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    print("\nNormalized Confusion Matrix (row-wise):")
    norm_df = pd.DataFrame(
        normalized_conf,
        index=categories,
        columns=categories
    ).applymap(lambda x: f"{x:.1%}")

    print(norm_df)
    norm_df.to_csv('FSNet/fsnet_confusion_matrix.csv', index_label='True\Predicted')