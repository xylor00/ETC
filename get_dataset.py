import pandas as pd
from IPhead import Get_headers
from ngram import create_plevel_feature


max_length = 100

all_flows_dict = Get_headers()

flow_sequences = []#存储数据包长度序列
IPheads = []#存储数据包头内容

merge_features = []#存储融合特征


for flow_key, flow_data in all_flows_dict.items():
    #读取每个流的类别
    label = flow_key[-1]
    
    #读取每个流的数据包长度序列
    pkt_length_sequence = flow_data['lengths']        
     
    #将长度序列截断或填充位等长，方便后续处理    
    if len(pkt_length_sequence) < max_length:
        pkt_length_sequence += [0] * (max_length - len(pkt_length_sequence))
    else:
        pkt_length_sequence = pkt_length_sequence[:max_length]
        
    pkt_length_sequence.append(label)
        
    flow_sequences.append(pkt_length_sequence)      
     
    #对每个流的IP包头数据进行n-gram处理
    IPhead_bytes = flow_data['byte']
    plevel_feature = create_plevel_feature(IPhead_bytes) 
    
    plevel_feature.append(label)
    
    IPheads.append(plevel_feature)      
    

df_flow = pd.DataFrame(flow_sequences)
df_flow.to_csv('dataset/flow_sequences.csv', index=False)

df_IPhead = pd.DataFrame(IPheads)
df_IPhead.to_csv('dataset/plevel_features.csv', index=False)