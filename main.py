import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transforms  # 图像变换工具
import multiprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

categories = ["Email", "Chat", "Streaming", "File Transfer", "Audio", "Video"]

# 自定义数据集类 --------------------------------------------------
class CSVDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        # 读取CSV数据
        self.data = pd.read_csv(csv_path, skiprows=1, header=None)
        
        # 分离特征和标签
        self.features = self.data.iloc[:, :-1].values.astype('float32')
        self.labels = self.data.iloc[:, -1].values
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = categories.index(self.labels[idx])
        
        if self.transform:
            x = self.transform(x)
            
        return x, y

# 模型定义 --------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # 超参数配置 --------------------------------------------------
    input_size = 256  
    hidden_size = 500
    num_classes = 6  # 6个类别
    num_epochs = 10
    batch_size = 100
    learning_rate = 0.001

    # 创建完整数据集实例
    full_dataset = CSVDataset('dataset/merge_features.csv')
    
    # 计算划分长度
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    
    # 随机划分数据集（设置随机种子保证可重复性）
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)

    # 模型初始化 --------------------------------------------------
    model = MLP(input_size, hidden_size, num_classes).to(device)  # 将模型移动到GPU（如果可用） 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    total_step = len(train_loader)  # 总批次数 = 训练集大小 / batch_size
    for epoch in range(num_epochs):  # 遍历每个训练周期
        for i, (flows, labels) in enumerate(train_loader):  # 遍历每批数据
            flows = flows.to(device)
            labels = labels.to(device)
        
            # 前向传播
            outputs = model(flows)  # 模型预测
            loss = criterion(outputs, labels)  # 计算损失
            
            # 反向传播和优化
            optimizer.zero_grad()  # 清空梯度缓存（重要！避免梯度累积）
            loss.backward()        # 反向传播计算梯度
            optimizer.step()       # 更新模型参数
            
            # 每2批打印训练信息
            if (i+1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item()}')

    # 测试时添加解码显示 --------------------------------------------------
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Test accuracy: {100 * correct / total}%')