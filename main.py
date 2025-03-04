import os
import pandas as pd
from traffic_augmentation import Traffic_Augmentation
def main():
    df = pd.read_csv("test.csv", skiprows=1, header=None)
    O = df.values.tolist()

    for i in O:
        newO = i[:-1]
        lable = i[-1]
        data = Traffic_Augmentation()
        aug_data = data.traffic_augmentation(newO, 1448)
        aug_flow = data.fit_data(aug_data, 20)
        aug_flow.append(lable)
        print(aug_flow)

if __name__ == '__main__':
    main()

