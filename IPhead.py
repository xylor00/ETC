import dpkt
import socket

max_byte_len = 12

categories = ["Chat", "Email"]
labels = [0, 1]


def gen_pkts(pcap, label):
    """按五元组分类存储数据包"""
    pkts = {}

    if pcap.datalink() != dpkt.pcap.DLT_EN10MB:
        print('unknown data link!')
        return pkts  # 返回空字典保持结构统一

    for _, buff in pcap:
        eth = dpkt.ethernet.Ethernet(buff)
        
        # 只处理IP层的TCP/UDP数据包
        if isinstance(eth.data, dpkt.ip.IP):
            ip = eth.data
            trans_proto = ip.data

            # 提取五元组要素
            if isinstance(trans_proto, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                # 转换IP地址为点分十进制格式
                src_ip = socket.inet_ntoa(ip.src)
                dst_ip = socket.inet_ntoa(ip.dst)
                
                # 获取协议类型名称
                proto_name = 'TCP' if isinstance(trans_proto, dpkt.tcp.TCP) else 'UDP'
                
                # 获取端口信息
                src_port = trans_proto.sport
                dst_port = trans_proto.dport
                
                flow_label = label# 原始标签
                label_name = categories[label]# 类别名称
                
                # 构造五元组标识键
                # (为了防止不同类型流量的五元组相同导致流量缺失，加上两个类别标识)
                flow_key = (
                    #五元组
                    proto_name,
                    src_ip, 
                    dst_ip,
                    src_port,
                    dst_port,
                    
                    #类别标识
                    flow_label,
                    label_name
                )

                # 初始化该五元组的存储列表
                if flow_key not in pkts:
                    pkts[flow_key] = {  
                        'packets': [],           # 数据包列表
                        'lengths': []            # 数据包长度序列
                    }
                
                # 添加数据包到对应五元组
                pkts[flow_key]['packets'].append(ip)
                pkts[flow_key]['lengths'].append(ip.len)
    return pkts


def closure(pkts_list):
    """整合多个pcap结果"""
    merged = {}
    for pkt_dict in pkts_list:
        for flow_key, flow_data in pkt_dict.items():
            if flow_key not in merged:
                merged[flow_key] = flow_data
            else:
                merged[flow_key]['packets'].extend(flow_data['packets'])
    return merged

def pkt2feature(all_flows):
    """将按五元组分类的流量数据转换为特征列表"""
    all_flows_dict = {}

    # 遍历每个五元组流
    for flow_key, flow_data in all_flows.items():
        # 获取该流的分类标签（字符串形式）
        all_flows_dict[flow_key] = {
            'byte': [], # 流量字节数据列表
            'lengths': [] # 数据包长度序列
        }
        all_flows_dict[flow_key]['lengths'] = flow_data['lengths']
        
        # 遍历当前流中的所有数据包
        for pkt in flow_data['packets']:
            # 将IP数据包序列化为原始字节流
            raw_byte = pkt.pack()
            
            # 提取前N个字节（带截断和填充）
            byte_features = []
            for x in range(min(len(raw_byte), max_byte_len)):
                byte_features.append(int(raw_byte[x]))  # 转换为整型
            
            # 填充不足部分
            byte_features.extend([0] * (max_byte_len - len(byte_features)))
            
            all_flows_dict[flow_key]['byte'].append(byte_features)
    
    return all_flows_dict


def get_headers():
    pkts_list = []

    chat = dpkt.pcap.Reader(open('testchat.pcap', 'rb'))
    chat_flows = gen_pkts(chat, 0)
    pkts_list.append(chat_flows)


    email = dpkt.pcap.Reader(open('testemail.pcap', 'rb'))
    email_flows = gen_pkts(email, 1)
    pkts_list.append(email_flows)

    all_flows = closure(pkts_list)

    all_flows_dict = pkt2feature(all_flows)
    return all_flows_dict

all_flows_dict = get_headers()
print(all_flows_dict)

"""
chat = dpkt.pcap.Reader(open('testchat.pcap', 'rb'))
chat_flows = gen_pkts(chat, 0)

i = 0
for key, data in chat_flows.items():
    if i < 1:
        for pkt in data['packets']:
            print(sum(pkt))
    i += 1
"""