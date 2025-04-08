import pandas as pd

flow_sequences = pd.read_csv('dataset/flow_sequences.csv', skiprows=1, header=None)
IPheads = pd.read_csv('dataset/IPheads.csv', skiprows=1, header=None)

flevel_features = flow_sequences.iloc[:, :-1].values.astype(int)
plevel_features = IPheads.iloc[:, :-1].values.astype(int)
labels = flow_sequences.iloc[:, -1].values

print(flevel_features)
print(plevel_features)