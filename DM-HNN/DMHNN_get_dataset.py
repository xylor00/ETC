import os
import dpkt
import socket
from collections import defaultdict
import csv
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# 根据论文修改的参数
MAX_PACKETS = 20       # 每个流取前20个包
MAX_BYTES = 40         # 每个包取前40字节
ONE_HOT_DIM = 256      # 字节值one-hot编码尺寸 (0-255)
POS_ENCODING_DIM = 256  # 位置编码维度 (必须与ONE_HOT_DIM相同)
categories = ["socialapp", "chat", "email", "file", "streaming", "VoIP"]
TARGET_DIM = 128       # 目标降维维度

# 预计算位置编码矩阵 (位置0-39, 维度256)
def get_positional_encoding(max_len=MAX_BYTES, d_model=POS_ENCODING_DIM):
    position_encoding = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            div_term = 10000 ** (i / d_model)
            position_encoding[pos, i] = math.sin(pos / div_term)
            if i + 1 < d_model:
                position_encoding[pos, i + 1] = math.cos(pos / div_term)
    return position_encoding

# 全局位置编码矩阵
POSITION_ENCODING_MATRIX = get_positional_encoding()

def byte_to_onehot(byte_val):
    """将字节值转换为one-hot编码 (256维)"""
    onehot = np.zeros(ONE_HOT_DIM)
    if 0 <= byte_val < ONE_HOT_DIM:
        onehot[byte_val] = 1
    return onehot

def apply_positional_encoding(byte_features):
    """
    应用位置编码到包字节特征
    """
    encoded = []
    for pos, byte_val in enumerate(byte_features):
        # 字节值转换为one-hot编码 (256维)
        byte_onehot = byte_to_onehot(byte_val)
        
        # 获取位置编码 (256维)
        pos_enc = POSITION_ENCODING_MATRIX[pos]
        
        # 将one-hot编码与位置编码相加 
        combined = byte_onehot + pos_enc
        encoded.append(combined)
    
    # 展平为1维数组 (40 * 256 = 10240维)
    return np.array(encoded).flatten()

def stream_packets(pcap, label):
    """流式生成器: 累积整个pcap的流, 处理完毕后统一过滤并yield"""
    flows = defaultdict(lambda: {'lengths': [], 'bytes': []})
    
    # 先完全解析整个pcap文件，累积所有流数据
    for _, buff in pcap:
        eth = dpkt.ethernet.Ethernet(buff)
        if not isinstance(eth.data, dpkt.ip.IP):
            continue
        
        ip = eth.data
        trans_proto = ip.data
        if not isinstance(trans_proto, (dpkt.tcp.TCP, dpkt.udp.UDP)):
            continue
        
        # 构造五元组键（包含label）
        flow_key = (
            'TCP' if isinstance(trans_proto, dpkt.tcp.TCP) else 'UDP',
            socket.inet_ntoa(ip.src),
            socket.inet_ntoa(ip.dst),
            trans_proto.sport,
            trans_proto.dport,
            label
        )
        
        # 提取应用层数据长度（流级特征）
        if isinstance(trans_proto, dpkt.tcp.TCP):
            tcp_header_len = trans_proto.off * 4
            app_data_len = ip.len - (ip.hl * 4) - tcp_header_len
        else:
            udp_header_len = 8
            app_data_len = ip.len - (ip.hl * 4) - udp_header_len
        
        # 确保长度至少为1（避免负值）
        pkt_len = max(app_data_len, 1)
        
        # 提取包级特征（前40字节）
        ip_packet = ip.pack()
        ip_header_length = ip.hl * 4
        
        # 取前40字节，不足则填充0
        raw_bytes = ip_packet[:MAX_BYTES]
        byte_features = list(raw_bytes) + [0] * (MAX_BYTES - len(raw_bytes))
        
        # 隐私保护：将IP地址和端口号置零
        # 1. 置零IP地址（源IP:12-15字节, 目的IP:16-19字节）
        if ip_header_length >= 20 and len(byte_features) >= 20:
            for i in range(12, 16):  # 源IP
                byte_features[i] = 0
            for i in range(16, 20):  # 目的IP
                byte_features[i] = 0
        
        # 2. 置零端口号（传输层头的前4字节）
        trans_start = ip_header_length
        if trans_start + 4 <= len(byte_features):
            for i in range(trans_start, trans_start + 4):
                byte_features[i] = 0
        
        # 累积到流字典
        flows[flow_key]['lengths'].append(pkt_len)
        flows[flow_key]['bytes'].append(byte_features)
    
    # 处理完整个pcap后，筛选并yield长度≥5的流
    for flow_key in list(flows.keys()):
        if len(flows[flow_key]['lengths']) >= 5:
            yield flow_key, flows[flow_key]
            del flows[flow_key]  # 释放内存

def length_to_onehot(length):
    """将包长转换为one-hot编码 (1501维)"""
    # 限制长度在0-1500范围内
    idx = min(max(int(length), 0), 1500)
    onehot = np.zeros(1501)
    onehot[idx] = 1
    return onehot

def process_all_pcaps(output_flow_csv, output_plevel_csv):
    """主处理函数: 收集数据后输出到CSV文件"""
    # 收集原始数据
    all_flevel_features = []  # 流级特征 (每个样本展平为30020维)
    all_plevel_features = []  # 包级特征 (每个样本展平为204800维)
    all_labels = []
    
    category_dirs = {
        #'socialapp': 'dataset_pcap/socialapp',
        #'chat': 'dataset_pcap/chat',
        #'email': 'dataset_pcap/email',
        #'file': 'dataset_pcap/file',
        #'streaming': 'dataset_pcap/streaming',
        #'VoIP': 'dataset_pcap/VoIP'
    }

    pcap_files = []
    for label, dir_path in category_dirs.items():
        if not os.path.exists(dir_path):
            print(f"warning: {dir_path} directory does not exist, skip", flush=True)
            continue
        
        for filename in os.listdir(dir_path):
            if filename.endswith('.pcap'):
                file_path = os.path.join(dir_path, filename)
                pcap_files.append((file_path, label))

    # 处理每个pcap文件
    for file_path, label in pcap_files:
        try:
            with open(file_path, 'rb') as f:
                pcap = dpkt.pcap.Reader(f)
                # 处理当前pcap的所有流
                for flow_key, flow_data in stream_packets(pcap, label):
                    # 流级特征处理 (包长序列)
                    lengths = flow_data['lengths'][:MAX_PACKETS]  # 取前20个包
                    # 不足20个包则填充0长度
                    if len(lengths) < MAX_PACKETS:
                        lengths += [0] * (MAX_PACKETS - len(lengths))
                    
                    # 将包长转换为one-hot编码并展平 (20*1501=30020维)
                    flevel_features = []
                    for length in lengths:
                        flevel_features.extend(length_to_onehot(length))
                    
                    # 包级特征处理 (前40字节 + one-hot + 位置编码)
                    byte_features = flow_data['bytes'][:MAX_PACKETS]  # 取前20个包
                    # 不足20个包则填充0字节
                    if len(byte_features) < MAX_PACKETS:
                        byte_features += [[0] * MAX_BYTES] * (MAX_PACKETS - len(byte_features))
                    
                    # 应用位置编码并展平 (20*40*256=204800维)
                    plevel_flat = []
                    for pkt_bytes in byte_features:
                        # 每个包: 40字节 → 10240维 (40*256)
                        encoded = apply_positional_encoding(pkt_bytes)
                        plevel_flat.extend(encoded)
                    
                    all_flevel_features.append(flevel_features)
                    all_plevel_features.append(plevel_flat)
                    all_labels.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}", flush=True)
            continue
        
        print(f"{file_path} finish", flush=True)
    
    # 转换为NumPy数组
    all_flevel_features = np.array(all_flevel_features)
    all_plevel_features = np.array(all_plevel_features)
    all_labels = np.array(all_labels)
    
    print(f"Original flow features shape: {all_flevel_features.shape}")
    print(f"Original packet features shape: {all_plevel_features.shape}")
    
    # ===== 先进行特征降维 =====
    print("Applying dimensionality reduction...")
    
    # 对流级特征降维
    flow_scaler = StandardScaler()
    flow_features_scaled = flow_scaler.fit_transform(all_flevel_features)
    flow_pca = PCA(n_components=TARGET_DIM)
    flow_reduced = flow_pca.fit_transform(flow_features_scaled)
    print(f"Flow features reduced to {flow_reduced.shape[1]} dimensions")
    
    # 对包级特征降维
    plevel_scaler = StandardScaler()
    plevel_features_scaled = plevel_scaler.fit_transform(all_plevel_features)
    plevel_pca = PCA(n_components=TARGET_DIM)
    plevel_reduced = plevel_pca.fit_transform(plevel_features_scaled)
    print(f"Packet features reduced to {plevel_reduced.shape[1]} dimensions")
    
    # 拼接降维后的特征
    X_reduced = np.hstack([flow_reduced, plevel_reduced])
    
    # ===== 再进行重采样 =====
    label_counter = Counter(all_labels)
    print("Original sample distribution:", label_counter)
    
    le = LabelEncoder()
    all_labels_numeric = le.fit_transform(all_labels)
    
    # 设置目标样本数
    target_samples = 500
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
        X_resampled, y_resampled = pipeline.fit_resample(X_reduced, all_labels_numeric)
    except ValueError as e:
        print(f"sampling error: {str(e)}")
        return
    
    # 拆分回原始特征格式
    flow_resampled = X_resampled[:, :TARGET_DIM]  # 流级特征
    plevel_resampled = X_resampled[:, TARGET_DIM:] # 包级特征
    
    # 转换回文本标签
    labels_resampled = le.inverse_transform(y_resampled)
    
    # 打乱顺序
    shuffled_idx = np.random.permutation(len(flow_resampled))
    flow_resampled = flow_resampled[shuffled_idx]
    plevel_resampled = plevel_resampled[shuffled_idx]
    labels_resampled = labels_resampled[shuffled_idx]
    
    # 写入CSV
    with open(output_flow_csv, 'w', newline='') as f_flow, \
         open(output_plevel_csv, 'w', newline='') as f_plevel:
        
        flow_writer = csv.writer(f_flow)
        plevel_writer = csv.writer(f_plevel)
        
        # 写CSV头
        flow_writer.writerow([f'feature_{i}' for i in range(TARGET_DIM)] + ['label'])
        plevel_writer.writerow([f'feature_{i}' for i in range(TARGET_DIM)] + ['label'])
        
        # 写入数据
        for flow_feat, plevel_feat, label in zip(flow_resampled, plevel_resampled, labels_resampled):
            flow_writer.writerow(list(flow_feat) + [label])
            plevel_writer.writerow(list(plevel_feat) + [label])
    
    print(f"Flow features saved to {output_flow_csv}, shape: {flow_resampled.shape}")
    print(f"Packet features saved to {output_plevel_csv}, shape: {plevel_resampled.shape}")
    
    """
    # 保存PCA模型供后续使用
    import joblib
    os.makedirs('DM-HNN/models', exist_ok=True)
    joblib.dump(flow_scaler, 'DM-HNN/models/flow_scaler.pkl')
    joblib.dump(flow_pca, 'DM-HNN/models/flow_pca.pkl')
    joblib.dump(plevel_scaler, 'DM-HNN/models/plevel_scaler.pkl')
    joblib.dump(plevel_pca, 'DM-HNN/models/plevel_pca.pkl')
    print("PCA models saved to DM-HNN/models/")
    """
    
if __name__ == '__main__':                 
    process_all_pcaps(
        output_flow_csv='DM-HNN/origin_flevel_features.csv',
        output_plevel_csv='DM-HNN/origin_plevel_features.csv'
    )