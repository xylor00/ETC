import os
import random
from scipy.stats import expon


class Traffic_Augmentation:
    def __init__(self, MAX_RTT = 0.01):
        self.MAX_RTT = MAX_RTT
        
    def traffic_augmentation(self, O, MSS):
        #O is a flow′s packet length sequence
        lengthflow = []
        interval = 0
        i = 0
        num = len(O) - 1
        delays = self.get_delay(len(O))
        while num >= i and int(O[i]) > 0:
            RTT = random.random() * self.MAX_RTT
            buf = 0
            while num >= i and RTT > 0:
                interval = delays[i]
                RTT -= interval
                buf += int(O[i]) - 40
                #some dataset have no head, that time no need to -40
                i += 1
            while buf > MSS:
                lengthflow.append(MSS + 40)
                buf -= MSS
            lengthflow.append(buf + 40)
        return lengthflow
    
    #randomly creat delay
    def get_delay(self, size):
        delays = []
        while len(delays) < size:
            t = random.random()
            if t < 0.1:
                delays.append(0.21)
            else:
                # loc, scale = 7e-06, 0.01094557476340694
                loc, scale = 1e-06, 0.00094557476340694
                t = expon.rvs(loc = loc, scale = scale, size = 1)
                delays.extend(t)
        return delays
    
    #make sure the data be the same length
    def fit_data(self, seq, max_l):
        lable = seq[-1]
        data = seq[:-1]
        if len(data) < max_l:
            while len(data) < max_l:
                data.append(0)
        else:
            data = seq[:max_l]
        data.append(lable)
        return data