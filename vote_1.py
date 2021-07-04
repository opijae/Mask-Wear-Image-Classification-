import torch
import random
import numpy as np
from pprint import pprint
import pandas as pd
import os
def softmax(x):

    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x
temp=None
csv_list=os.listdir('./output')
for csv in csv_list:
    cvv_path=os.path.join('./output',csv)
    csv_data=np.loadtxt(cvv_path,delimiter=',',dtype=np.float64)
    if temp is None:
        temp=softmax(csv_data)
    else:
        temp*=softmax(csv_data)

info_path = os.path.join('/opt/ml/input/data/eval', 'info.csv')
info = pd.read_csv(info_path)
info['ans'] = np.argmax(temp, axis=1)
info.to_csv('final_with_reg_55_25.csv', index=False)