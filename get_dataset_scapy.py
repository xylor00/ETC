import os
import dpkt
import socket
from collections import defaultdict, Counter
import csv
import numpy as np
from ngram import create_plevel_feature
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from scapy.all import PcapReader, IP, TCP, UDP
import gc

max_byte_len = 12
#categories = ["socialapp", "chat", "email", "file", "streaming", "VoIP"]
#categories = ["socialapp", "chat", "email", "file", "streaming", "web"]
#categories = ["Benign", "Malware"]
categories = ["chat", "file", "streaming", "VoIP", "C2"]

def stream_packets(file_path, label):
    """流式生成器: 累积整个pcap的流, 处理完毕后统一过滤并yield"""
    flows = defaultdict(lambda: {'lengths': [], 'byte': []})
    
    try:
        # rdpcap 会自动处理 pcap 和 pcapng 格式
        pkts = PcapReader(file_path)
    except Exception as e:
        print(f"Error reading {file_path} with Scapy: {e}")
        return []
    
    # 先完全解析整个文件，累积所有流数据
    for pkt in pkts:
        # 确保包含 IP 层和传输层（TCP/UDP）
        if IP in pkt and (TCP in pkt or UDP in pkt):
            ip_layer = pkt[IP]
            trans_proto = 'TCP' if TCP in pkt else 'UDP'
            trans_layer = pkt[TCP] if TCP in pkt else pkt[UDP]
            
            # 构造五元组键
            flow_key = (
                trans_proto,
                ip_layer.src,
                ip_layer.dst,
                trans_layer.sport,
                trans_layer.dport,
                label
            )
        
        # 提取特征
        # 提取应用层数据长度        
        app_data_len = len(trans_layer.payload)
        pkt_len = max(app_data_len, 1)# 确保包长度至少为1
        
        # 提取 IP 头的前 12 字节
        # bytes(ip_layer) 获取原始 IP 报文，切片取前 12 字节
        raw_ip_header = bytes(ip_layer)[:12]
        byte_features = list(raw_ip_header) + [0] * (max_byte_len - len(raw_ip_header))
    
        # 累积到流字典
        flows[flow_key]['lengths'].append(pkt_len)
        flows[flow_key]['byte'].append(byte_features)
    
    # 筛选长度 ≥ 5 的流
    valid_flows = []
    for flow_key, data in flows.items():
        if len(data['lengths']) >= 5:
            valid_flows.append((flow_key, data))
            
    return valid_flows

def process_all_pcaps(output_flow_csv, output_plevel_csv, max_length=100):
    """主处理函数: 收集数据后重采样"""
    # 收集原始数据
    all_flow_features = []
    all_plevel_features = []
    all_labels = []
    
    category_dirs = {
        #'socialapp': 'VPN-NonVPN/socialapp',
        #'chat': 'VPN-NonVPN/chat',
        #'email': 'VPN-NonVPN/email',
        #'file': 'VPN-NonVPN/file',
        #'streaming': 'VPN-NonVPN/streaming',
        #'VoIP': 'VPN-NonVPN/VoIP',
        #'web': 'Tor-NonTor/web',
        #'Benign': 'USTC-TFC/Benign',
        #'Malware': 'USTC-TFC/Malware' 
        'C2': 'VNAT/C2',
        'chat': 'VNAT/chat',
        'file': 'VNAT/file',
        'streaming': 'VNAT/streaming',
        'VoIP': 'VNAT/VoIP'             
    }

    pcap_files = []
    for label, dir_path in category_dirs.items():
        if not os.path.exists(dir_path):
            print(f"warning: {dir_path} directory does not exist, skip", flush=True)
            continue
        
        for filename in os.listdir(dir_path):
            if filename.endswith(('.pcap', '.pcapng')):
                file_path = os.path.join(dir_path, filename)
                pcap_files.append((file_path, label))

    # 处理每个pcap文件
    for file_path, label in pcap_files:
        
        try:
            flow_list = stream_packets(file_path, label)
        
            for flow_key, flow_data in flow_list:
                # 长度序列提取（填充/截断到max_length）
                lengths = flow_data['lengths'][:max_length] + [0] * (max_length - len(flow_data['lengths']))
                all_flow_features.append(lengths)
                
                # 包特征处理
                plevel_feature = create_plevel_feature(flow_data['byte'])
                all_plevel_features.append(plevel_feature)
                
                all_labels.append(label)
            
            #及时释放当前文件的临时内存
            del flow_list
            gc.collect()
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}", flush=True)
            continue
        
        print(f"{file_path} finish", flush=True)
        
    label_counter = Counter(all_labels)
    print("\n--- Original dataset distribution ---")
    for cat, count in label_counter.items():
        print(f"category {cat}: {count} samples")
    
    
    # 平衡数据集
    le = LabelEncoder()
    all_labels_numeric = le.fit_transform(all_labels)
    
    # 创建复合特征矩阵（保持结构分离）
    X_compound = np.hstack([
        np.array(all_flow_features, dtype=np.float64),   # shape: (n, 100)
        np.array(all_plevel_features)  # shape: (n, 165)
    ])
    
    # 初始化SMOTE（自动处理所有类别到target_samples个样本）
    # 设置目标样本数
    target_samples = 1000
    valid_categories = []

    # 筛选有效类别（至少2个样本）
    for cat in categories:
        if label_counter.get(cat, 0) >= 2:
            valid_categories.append(cat)
    
    # 配置组合采样器
    over = SMOTE(
        sampling_strategy={le.transform([cat])[0]: target_samples 
        for cat in valid_categories 
        if label_counter[cat] < target_samples
    },
    k_neighbors=5
    )

    under = RandomUnderSampler(
        sampling_strategy={le.transform([cat])[0]: target_samples 
        for cat in valid_categories 
        if label_counter[cat] > target_samples
    })

    pipeline = Pipeline([('over', over), ('under', under)])

    try:
        X_resampled, y_resampled = pipeline.fit_resample(X_compound, all_labels_numeric)
    except ValueError as e:
        print(f"sampling error: {str(e)}")
        return
    
    # 拆分回原始特征格式
    flow_resampled = X_resampled[:, :max_length]  # 前100列是flow特征
    plevel_resampled = X_resampled[:, max_length:] # 后165列是plevel特征
    
    # Flow特征特殊处理
    flow_resampled = np.round(flow_resampled).astype(int)  # 四舍五入取整
    flow_resampled = np.clip(flow_resampled, 0, 1448)     # 限制范围
    
    # 转换回文本标签
    labels_resampled = le.inverse_transform(y_resampled)
    
    # 打乱顺序
    shuffled_idx = np.random.permutation(len(flow_resampled))
    flow_resampled = flow_resampled[shuffled_idx]
    plevel_resampled = plevel_resampled[shuffled_idx]
    labels_resampled = labels_resampled[shuffled_idx]
    

    # 写入CSV（保持原有格式不变）
    with open(output_flow_csv, 'w', newline='') as f_flow, \
         open(output_plevel_csv, 'w', newline='') as f_plevel:
        
        flow_writer = csv.writer(f_flow)
        plevel_writer = csv.writer(f_plevel)
        
        # 写CSV头
        flow_writer.writerow([f'feature_{i}' for i in range(max_length)] + ['label'])
        plevel_writer.writerow([f'feature_{i}' for i in range(165)] + ['label'])
        
        # 写入数据
        for flow_feat, plevel_feat, label in zip(flow_resampled, plevel_resampled, labels_resampled):
            flow_writer.writerow(list(flow_feat) + [label])
            plevel_writer.writerow(list(plevel_feat) + [label])

if __name__ == '__main__':                 
    process_all_pcaps(
        output_flow_csv='features/flow_sequences.csv',
        output_plevel_csv='features/plevel_features.csv'
    )