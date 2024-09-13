import numpy as np
import pandas as pd

'''
dataset = pd.read_csv('health.txt',sep='|',usecols=['tweet'])
print(dataset);
dataset.to_csv("temp.csv",index=False)
'''

l = [1,1,1,0,0,0,1]
coount = np.count_nonzero(l)
print(coount)
