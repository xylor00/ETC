import os
import dpkt
import socket
from collections import defaultdict
import csv
from ngram import create_plevel_feature

max_byte_len = 12
min_tcp_len = 40
min_udp_len = 28
categories = ["Email", "Chat", "Streaming", "File Transfer", "VoIP", "P2P"]

def stream_packets(pcap, label):
    """流式生成器: 累积整个pcap的流, 处理完毕后统一过滤并yield"""
    flows = defaultdict(lambda: {'lengths': [], 'byte': []})
    
    # 先完全解析整个pcap文件，累积所有流数据
    for _, buff in pcap:
        eth = dpkt.ethernet.Ethernet(buff)
        if not isinstance(eth.data, dpkt.ip.IP):
            continue
        
        ip = eth.data
        trans_proto = ip.data
        if not isinstance(trans_proto, (dpkt.tcp.TCP, dpkt.udp.UDP)):
            continue
        
        min_len = min_tcp_len if isinstance(trans_proto, dpkt.tcp.TCP) else min_udp_len
        if ip.len < min_len:#排除掉长度小于包头的数据包（此类数据包可能已损坏）
            continue
        
        # 构造五元组键（包含label）
        flow_key = (
            'TCP' if isinstance(trans_proto, dpkt.tcp.TCP) else 'UDP',
            socket.inet_ntoa(ip.src),
            socket.inet_ntoa(ip.dst),
            trans_proto.sport,
            trans_proto.dport,
            label  # 标签作为五元组的一部分
        )
        
        # 提取特征
        pkt_len = max(ip.len - min_len, 1)
        raw_byte = ip.pack()[:max_byte_len]
        byte_features = list(raw_byte) + [0] * (max_byte_len - len(raw_byte))
        
        # 累积到流字典
        flows[flow_key]['lengths'].append(pkt_len)
        flows[flow_key]['byte'].append(byte_features)
    
    # 处理完整个pcap后，筛选并yield长度≥5的流
    for flow_key in list(flows.keys()):  # 避免字典修改异常
        if len(flows[flow_key]['lengths']) >= 5:
            yield flow_key, flows[flow_key]
            del flows[flow_key]  # 释放内存

def process_all_pcaps(output_flow_csv, output_plevel_csv, max_length=100):
    """主处理函数: 逐个pcap处理, 写入CSV"""
    with open(output_flow_csv, 'w', newline='') as f_flow, \
         open(output_plevel_csv, 'w', newline='') as f_plevel:
        
        flow_writer = csv.writer(f_flow)
        plevel_writer = csv.writer(f_plevel)
        
        # 写入CSV头
        flow_writer.writerow([f'feature_{i}' for i in range(max_length)] + ['label'])
        plevel_writer.writerow([f'feature_{i}' for i in range(165)] + ['label'])

        # 配置类别目录（可根据实际路径修改）
        category_dirs = {
            'Chat': 'dataset_pcap/Chat',
            'Email': 'dataset_pcap/Email',
            'Streaming': 'dataset_pcap/Streaming',
            'File Transfer': 'dataset_pcap/File Transfer',
            'VoIP': 'dataset_pcap/VoIP',
            'P2P': 'dataset_pcap/P2P'
        }

        # 收集所有pcap文件路径
        pcap_files = []
        for label, dir_path in category_dirs.items():
            if not os.path.exists(dir_path):
                print(f"warning: {dir_path} directory does not exist, skip")
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
                        # 流级特征（填充/截断到max_length）
                        lengths = flow_data['lengths'][:max_length] + [0] * (max_length - len(flow_data['lengths']))
                        flow_writer.writerow(lengths + [label])
                        
                        # 包级特征
                        plevel_feature = create_plevel_feature(flow_data['byte'])
                        plevel_writer.writerow(plevel_feature + [label])
            except Exception as e:
                print(f"Error {str(e)} occurred while processing file {file_path}")
                continue
            
            print(f"{file_path} finish")

            

if __name__ == '__main__':
    # 处理所有pcap文件并生成CSV                    
    process_all_pcaps(
        output_flow_csv='features/flow_sequences.csv',
        output_plevel_csv='features/plevel_features.csv'
    )