import math  # 需要导入math库

p_level_pkt_num = 5

def append_pkt_ngram(pkt, n):
    hex_pkt = [f"{byte:02x}" for byte in pkt]
    
    # 计算理论最大值和对数最大值
    max_val = (16 ** (2 * n)) - 1
    log_max = math.log(1 + max_val)  # 避免log(0)
    
    ngram_pkt = []
    for i in range(len(hex_pkt) - n + 1):
        ngram = "".join(hex_pkt[i:i+n])
        ngram_dec = int(ngram, 16)
        
        # 对数压缩后归一化
        log_val = math.log(1 + ngram_dec)
        normalized = log_val / log_max
        normalized = 2 * normalized - 1       # 新范围 [-1,1]
        ngram_pkt.append(normalized)
    
    return ngram_pkt

def create_plevel_feature(IPhead_bytes):
    plevel_feature = []
    
    # 确定实际数据包数量
    actual_pkt_num = min(len(IPhead_bytes), p_level_pkt_num)
    
    # 处理实际存在的数据包
    for i in range(actual_pkt_num):
        pkt = IPhead_bytes[i]
        
        # 原始字节归一化（保持线性）
        normalized_bytes = [(byte/255.0)*4 - 2 for byte in pkt]
        plevel_feature.extend(normalized_bytes)
        
        # 添加对数压缩后的2-gram和3-gram
        plevel_feature.extend(append_pkt_ngram(pkt, 2))
        plevel_feature.extend(append_pkt_ngram(pkt, 3))
    
    # 处理不足的包（零填充）
    for i in range(actual_pkt_num, p_level_pkt_num):
        # 创建与第一个包相同长度的零填充包
        # 如果还没有包，则使用空列表（长度0）
        zero_pkt = [0] * (len(IPhead_bytes[0]) if IPhead_bytes else [0])
        
        # 原始字节归一化（全零）
        normalized_bytes = [0.0] * len(zero_pkt)  # 归一化后全为-2
        plevel_feature.extend(normalized_bytes)
        
        # 添加零填充的2-gram和3-gram
        plevel_feature.extend([-1.0] * (len(zero_pkt) - 1))  # 2-gram填充
        plevel_feature.extend([-1.0] * (len(zero_pkt) - 2))  # 3-gram填充
    
    return plevel_feature