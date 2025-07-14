import os
import dpkt
import socket
from collections import defaultdict
import csv
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import numpy as np
from collections import Counter
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

categories = ["socialapp", "chat", "email", "file", "streaming", "VoIP"]
#categories = ["socialapp", "chat", "email", "file", "streaming", "web"]
#categories = ["Benign", "Malware"]

# 提取的每个流的最大字节数
MAX_FLOW_BYTES = 784

def stream_packets(pcap, label):
    """
    流式生成器: 累积整个pcap的流，处理完毕后统一过滤并yield。
    修改为收集每个流的原始数据包字节序列。
    """
    # flows 现在将存储每个流的完整数据包字节序列
    flows = defaultdict(lambda: {'packets_bytes': []})

    # 先完全解析整个pcap文件，累积所有流数据
    for _, buff in pcap:
        # buff 是原始数据包的完整字节序列
        eth = dpkt.ethernet.Ethernet(buff)
        if not isinstance(eth.data, dpkt.ip.IP):
            continue

        ip = eth.data
        trans_proto = ip.data
        if not isinstance(trans_proto, (dpkt.tcp.TCP, dpkt.udp.UDP)):
            continue

        # 构造五元组键（包含label），用于识别唯一的流
        flow_key = (
            'TCP' if isinstance(trans_proto, dpkt.tcp.TCP) else 'UDP',
            socket.inet_ntoa(ip.src),
            socket.inet_ntoa(ip.dst),
            trans_proto.sport,
            trans_proto.dport,
            label
        )

        # 提取特征：直接使用整个原始数据包的字节。
        # 累积到流字典
        flows[flow_key]['packets_bytes'].append(buff)

    # 处理完整个pcap后，筛选并yield长度≥5的流
    for flow_key in list(flows.keys()):  # 遍历字典的键，避免在迭代时修改字典
        if len(flows[flow_key]['packets_bytes']) >= 5: # 确保流至少包含5个数据包
            yield flow_key, flows[flow_key]
            del flows[flow_key]  # 释放内存，避免内存占用过高

def process_all_pcaps(output_784byte_csv):
    """
    主处理函数: 收集数据后进行重采样。
    提取并处理每个流的前784个字节。
    """
    # 用于存储所有流的 784 字节特征
    all_784byte_features = []
    all_labels = [] # 用于存储所有流的标签

    category_dirs = {
        'socialapp': 'Tor-NonTor/socialapp',
        'chat': 'Tor-NonTor/chat',
        'email': 'Tor-NonTor/email',
        'file': 'Tor-NonTor/file',
        'streaming': 'Tor-NonTor/streaming',
        'VoIP': 'Tor-NonTor/VoIP',
        #'web': 'Tor-NonTor/web',
        #'Benign': 'USTC/Benign',
        #'Malware': 'USTC/Malware'   
    }

    pcap_files = []
    # 遍历每个类别目录，找到所有的 .pcap 文件
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
                    # 获取流中所有数据包的原始字节并拼接成一个长字节序列
                    combined_bytes = b''.join(flow_data['packets_bytes'])

                    # 截断或填充到 MAX_FLOW_BYTES (784)
                    if len(combined_bytes) >= MAX_FLOW_BYTES:
                        flow_784_bytes = combined_bytes[:MAX_FLOW_BYTES]
                    else:
                        # 如果字节数不足，用空字节 (b'\x00') 填充
                        flow_784_bytes = combined_bytes + b'\x00' * (MAX_FLOW_BYTES - len(combined_bytes))

                    # 将字节序列转换为整数列表（每个字节一个整数，范围 0-255）
                    byte_features_list = list(flow_784_bytes)
                    all_784byte_features.append(byte_features_list) # 收集特征
                    all_labels.append(label) # 收集标签
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}", flush=True)
            continue

        print(f"{file_path} finish", flush=True)

    label_counter = Counter(all_labels)
    print("Original sample distribution:", label_counter)

    # 平衡数据集
    le = LabelEncoder()
    all_labels_numeric = le.fit_transform(all_labels)

    # 将 784 字节特征列表转换为 numpy 数组，作为复合特征矩阵
    # 确保数据类型为浮点数，以便 SMOTE 可以处理
    X_compound = np.array(all_784byte_features, dtype=np.float64)

    # 设置目标样本数，以平衡各类样本数量(考虑到特征的长度，减少采样数量)
    target_samples = 1000
    valid_categories = []

    # 筛选有效类别（至少2个样本才能进行SMOTE）
    for cat in categories:
        if label_counter.get(cat, 0) >= 2:
            valid_categories.append(cat)

    # 配置组合采样器 (SMOTE 过采样少数类，RandomUnderSampler 欠采样多数类)
    over = SMOTE(
        sampling_strategy={le.transform([cat])[0]: target_samples
        for cat in valid_categories
        if label_counter[cat] < target_samples
    },
    k_neighbors=5 # 用于SMOTE的最近邻数量
    )

    under = RandomUnderSampler(
        sampling_strategy={le.transform([cat])[0]: target_samples
        for cat in valid_categories
        if label_counter[cat] > target_samples
    })

    # 创建一个管道，先过采样再欠采样
    pipeline = Pipeline([('over', over), ('under', under)])

    try:
        # 执行重采样
        X_resampled, y_resampled = pipeline.fit_resample(X_compound, all_labels_numeric)
    except ValueError as e:
        print(f"采样错误: {str(e)}", flush=True)
        return

    flow_784_resampled = X_resampled

    # SMOTE 生成的特征是浮点数，需要四舍五入并转换回整数
    flow_784_resampled = np.round(flow_784_resampled).astype(int)
    # 字节值范围是 0-255，进行裁剪以确保值的有效性
    flow_784_resampled = np.clip(flow_784_resampled, 0, 255)

    # 将数值标签转换回原始文本标签
    labels_resampled = le.inverse_transform(y_resampled)

    # 打乱顺序，增加数据集的随机性
    shuffled_idx = np.random.permutation(len(flow_784_resampled))
    flow_784_resampled = flow_784_resampled[shuffled_idx]
    labels_resampled = labels_resampled[shuffled_idx]

    # 写入CSV文件
    with open(output_784byte_csv, 'w', newline='') as f_784byte:
        writer_784byte = csv.writer(f_784byte)

        # 写CSV头，列名表示字节位置
        writer_784byte.writerow([f'byte_{i}' for i in range(MAX_FLOW_BYTES)] + ['label'])

        # 写入数据行
        for byte_feat, label in zip(flow_784_resampled, labels_resampled):
            writer_784byte.writerow(list(byte_feat) + [label])


if __name__ == '__main__':
    process_all_pcaps(
        output_784byte_csv='1dcnn/784byte.csv'
    )
