# **基于融合特征的加密流量分类**

1. 运行get_dataset.py，从pcap文件中提取可用数据，生成包级别特征
2. 运行byoltrainer.py，使用数据包长度序列对GRU网络进行训练，生成流级别特征
3. 运行get_merge_feature.py，生成融合特征
4. 运行main.py

