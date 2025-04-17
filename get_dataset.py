import pandas as pd
from IPhead import Get_headers
from ngram import create_plevel_feature
import csv
from functools import partial

max_length = 100

# 获取所有流数据
all_flows_dict = Get_headers()

# 直接打开文件句柄，逐步写入数据
# 使用csv模块替代Pandas，减少内存占用
flow_csv = open('dataset/flow_sequences.csv', 'w', newline='', buffering=100)
iphead_csv = open('dataset/plevel_features.csv', 'w', newline='', buffering=100)

flow_writer = csv.writer(flow_csv)
iphead_writer = csv.writer(iphead_csv)

# 写入CSV头部（无列名）
f_feature_columns = [f'feature_{i}' for i in range(256)]
i_feature_columns = [f'feature_{i}' for i in range(165)]
flow_writer.writerow(f_feature_columns + ['label'])
iphead_writer.writerow(i_feature_columns + ['label'])

# 分块处理每个流
for flow_key, flow_data in all_flows_dict.items():
    # 处理流级特征（包长度序列）
    label = flow_key[-1]
    pkt_length_sequence = flow_data['lengths']
    
    # 填充/截断长度序列
    if len(pkt_length_sequence) < max_length:
        pkt_length_sequence += [0] * (max_length - len(pkt_length_sequence))
    else:
        pkt_length_sequence = pkt_length_sequence[:max_length]
    pkt_length_sequence.append(label)
    
    # 直接写入流级特征到文件
    flow_writer.writerow(pkt_length_sequence)
    
    # 处理包级特征（IP头n-gram）
    IPhead_bytes = flow_data['byte']
    plevel_feature = create_plevel_feature(IPhead_bytes)
    plevel_feature.append(label)
    
    # 直接写入包级特征到文件
    iphead_writer.writerow(plevel_feature)

# 关闭文件句柄
flow_csv.close()
iphead_csv.close()