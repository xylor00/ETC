import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time

FLOW_DIM = 128  # 流级特征维度 (来自预处理PCA)
PKT_DIM = 128   # 包级特征维度 (来自预处理PCA)
HIDDEN_DIM = 256  # 修改为256维
categories = ["socialapp", "chat", "email", "file", "streaming", "VoIP"]
#categories = ["socialapp", "chat", "email", "file", "streaming", "web"]
#categories = ["Benign", "Malware"]

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 自定义数据集类
class DualModeDataset(Dataset):
    def __init__(self, flow_csv, plevel_csv):
        # 读取流级特征CSV
        flow_df = pd.read_csv(flow_csv)
        self.flow_features = flow_df.iloc[:, :-1].values.astype(np.float32)
        self.labels = flow_df.iloc[:, -1].values
        
        # 读取包级特征CSV
        plevel_df = pd.read_csv(plevel_csv)
        self.plevel_features = plevel_df.iloc[:, :-1].values.astype(np.float32)
        
        # 标签编码
        self.label_map = {label: idx for idx, label in enumerate(np.unique(self.labels))}
        self.encoded_labels = np.array([self.label_map[label] for label in self.labels])
        
        print(f"Loaded dataset with {len(self.flow_features)} samples", flush=True)
        print(f"Flow features shape: {self.flow_features.shape}", flush=True)
        print(f"Packet-level features shape: {self.plevel_features.shape}", flush=True)
        print(f"Label distribution: {np.unique(self.labels, return_counts=True)}", flush=True)
        
        # 保存标签映射关系
        self.reverse_label_map = {idx: label for label, idx in self.label_map.items()}

    def __len__(self):
        return len(self.flow_features)

    def __getitem__(self, idx):
        return {
            'flow': self.flow_features[idx],
            'plevel': self.plevel_features[idx],
            'label': self.encoded_labels[idx]
        }

# 改进的GRU路径 (流级特征处理)
class FlowGRUPath(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(FlowGRUPath, self).__init__()
        # 添加线性层将输入维度转换为hidden_dim
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        # 开关机制
        self.switch_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.fc = nn.Linear(hidden_dim, 256)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x):
        # 将输入转换为hidden_dim维度
        x_transformed = self.input_fc(x)
        
        # 添加时间步维度
        x_transformed = x_transformed.unsqueeze(1)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x_transformed.size(0), self.hidden_dim).to(device)
        
        # GRU处理
        out, hn = self.gru(x_transformed, h0)
        gru_output = out[:, -1, :]
        
        # 应用开关机制 
        combined = torch.cat([x_transformed.squeeze(1), gru_output], dim=1)
        switch_param = self.switch_gate(combined)
        
        # 最终输出
        output = (1 - switch_param) * x_transformed.squeeze(1) + switch_param * gru_output
        
        # 全连接层进一步处理
        return self.fc(output)

# SAE路径 (包级特征处理)
class PacketSAEPath(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 256]):
        super(PacketSAEPath, self).__init__()
        # 构建编码器
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.ReLU(True))
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 构建解码器 (用于无监督预训练)
        decoder_layers = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            decoder_layers.append(nn.ReLU(True))
        decoder_layers.append(nn.Linear(hidden_dims[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # 最终特征提取层 - 256
        self.fc = nn.Linear(hidden_dims[0], 256)

    def forward(self, x):
        # 编码
        encoded = self.encoder(x)
        
        # 解码 (仅用于预训练阶段)
        decoded = self.decoder(encoded)
        
        # 最终特征提取
        return self.fc(encoded), decoded

# 双模式特征提取模型
class DualFeatureExtractor(nn.Module):
    def __init__(self, flow_dim, pkt_dim, hidden_dim):
        super(DualFeatureExtractor, self).__init__()
        self.flow_path = FlowGRUPath(flow_dim, hidden_dim)
        self.pkt_path = PacketSAEPath(pkt_dim, [256, 256])
        
    def forward(self, flow_input, pkt_input):
        flow_feature = self.flow_path(flow_input)
        pkt_feature, decoded = self.pkt_path(pkt_input)
        return flow_feature, pkt_feature, decoded

# 训练函数
def train_model(model, dataloader, criterion, optimizer, epochs):
    model.train()
    start_time = time.time()
    
    # 早停参数
    best_total_loss = 1000
    patience = 10
    no_improve_epochs = 0
    stop_training = False
    
    for epoch in range(epochs):
        # 检查早停条件
        if stop_training:
            print(f"Early stopping at epoch {epoch}", flush=True)
            print(f"get model at epoch {epoch-patience}", flush=True)
            break    
        
        total_loss = 0.0
        
        for batch in dataloader:
            flow_input = batch['flow'].to(device)
            pkt_input = batch['plevel'].to(device)
            
            # 前向传播
            flow_feature, pkt_feature, decoded = model(flow_input, pkt_input)
            
            # 计算重建损失 (仅对包级路径)
            loss = criterion(decoded, pkt_input)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Time: {time.time()-start_time:.2f}s', flush=True)
        
        # 早停判断
        if total_loss < best_total_loss:
            best_total_loss = total_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'model/best_model.pth')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                stop_training = True
    
    print(f'Training completed in {time.time()-start_time:.2f} seconds', flush=True)
    return model

# 保存特征为CSV文件
def save_features_to_csv(features, labels, reverse_label_map, filename):
    # 转换标签索引为原始标签
    str_labels = [reverse_label_map[label] for label in labels]
    
    # 创建DataFrame
    feature_columns = [f'feature_{i}' for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feature_columns)
    df['label'] = str_labels
    
    # 保存为CSV
    df.to_csv(filename, index=False)
    print(f"Features saved to {filename} with shape {features.shape}", flush=True)

# 主函数
def main():
    # 创建数据集
    dataset = DualModeDataset(
        flow_csv='DM-HNN/origin_flevel_features.csv',
        plevel_csv='DM-HNN/origin_plevel_features.csv'
    )
    
    # 创建数据加载器
    batch_size = 512
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = DualFeatureExtractor(
        flow_dim=FLOW_DIM,
        pkt_dim=PKT_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()  # 用于SAE重建损失
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # 训练模型
    print("Starting training...", flush=True)
    model = train_model(model, dataset_loader, criterion, optimizer, epochs=200)
    
    # 保存模型
    torch.save(model.state_dict(), 'DM-HNN/dual_feature_extractor.pth')
    print("Model saved to DM-HNN/dual_feature_extractor.pth", flush=True)
    
    # 提取整个数据集的特征
    model.eval()
    all_flow_features = []
    all_pkt_features = []
    all_labels = []
    
    # 创建整个数据集的数据加载器
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in full_loader:
            flow_input = batch['flow'].to(device)
            pkt_input = batch['plevel'].to(device)
            labels = batch['label'].numpy()
            
            flow_feature, pkt_feature, _ = model(flow_input, pkt_input)
            
            all_flow_features.append(flow_feature.cpu().numpy())
            all_pkt_features.append(pkt_feature.cpu().numpy())
            all_labels.append(labels)
    
    # 合并特征和标签
    flow_features = np.vstack(all_flow_features)
    pkt_features = np.vstack(all_pkt_features)
    all_labels = np.concatenate(all_labels)
    
    print(f"Extracted flow features shape: {flow_features.shape}", flush=True)
    print(f"Extracted packet features shape: {pkt_features.shape}", flush=True)
    
    # 保存为CSV文件
    save_features_to_csv(
        flow_features, 
        all_labels, 
        dataset.reverse_label_map, 
        'DM-HNN/final_flevel_features.csv'
    )
    
    save_features_to_csv(
        pkt_features, 
        all_labels, 
        dataset.reverse_label_map, 
        'DM-HNN/final_plevel_features.csv'
    )

if __name__ == '__main__':
    main()