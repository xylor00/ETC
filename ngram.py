# 示例数据转换验证
flow = [
    [69, 0, 0, 50, 18, 223, 0, 0, 1, 17, 81, 207],
    [69, 0, 0, 113, 125, 95, 0, 0, 128, 17, 72, 245],
    [69, 0, 0, 117, 56, 29, 0, 0, 128, 17, 142, 27],
    [69, 0, 0, 50, 35, 41, 0, 0, 1, 17, 63, 110],
    [69, 0, 0, 50, 35, 42, 0, 0, 1, 17, 63, 109]
]

def append_pkt_ngram(pkt, n):
    # 步骤1：将每个整数转为2位十六进制字符串（小写补零）
    hex_pkt = [f"{byte:02x}" for byte in pkt]  # 如69 → '45'
    
    ngram_pkt = []
    # 步骤2：生成滑动窗口拼接的n-gram
    for i in range(len(hex_pkt) - n + 1):
        # 拼接连续n个十六进制字符串
        ngram = "".join(hex_pkt[i:i+n])
        
        #将十六进制字符串转换为十进制并加入原流量中
        ngram_dec = int(ngram, 16)
        ngram_pkt.append(ngram_dec)
    
    return ngram_pkt

def create_plevel_feature(flow):
    #初始化包级别特征
    plevel_feature = []
    i = 0
    
    for pkt in flow:
        plevel_feature.append(pkt)
        
        #每个数据包进行2-gram, 3-gram特征提取
        gram_pkt_2 = append_pkt_ngram(pkt, 2)
        gram_pkt_3 = append_pkt_ngram(pkt, 3)
        
        plevel_feature[i].extend(gram_pkt_2)
        plevel_feature[i].extend(gram_pkt_3)
        
        i += 1
        
    return plevel_feature