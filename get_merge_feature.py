import pandas as pd
import numpy as np

f_features = pd.read_csv('dataset/flevel_features.csv', skiprows=1, header=None)
p_features = pd.read_csv('dataset/plevel_features.csv', skiprows=1, header=None)

flevel_features = f_features.iloc[:, :-1].values
plevel_features = p_features.iloc[:, :].values

merge_features = []#存储融合特征

#将数据包长度序列和包级别特征融合，并给三种特征都加上标签

merge_features = np.concatenate((flevel_features, plevel_features), axis=1)

df_merge = pd.DataFrame(merge_features)
df_merge.to_csv('dataset/merge_features.csv', index=False)