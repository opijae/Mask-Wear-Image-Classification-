import random
import numpy as np
import pandas as pd
import os
def softmax(x):

    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x
temp=pd.DataFrame()
csv_list=os.listdir('./output')
csv_list.sort()
for i,csv_file in enumerate(csv_list):
    if csv_file=='output_real_2.csv' or csv_file=='output_change_4to5.csv' or csv_file=='output_change_1to2.csv':
        continue
    csv_path=os.path.join('./output',csv_file)
    print(csv_path)
    info = pd.read_csv(csv_path)
    temp[csv_file[4:]]=info['ans']
def check(temp):
    '''
    각 csv 파일 체크
    '''

    if temp is None:
        temp[0]=pd.read_csv(os.path.join('./output','output_real_4_best.csv'))['ans']
        temp[1]=pd.read_csv('output_xcept+res.csv')['ans']
        temp[2]=pd.read_csv('output_res_kfold.csv')['ans']

    print(temp.head(10))
    temp=temp.to_numpy()
    ans=[]
    for arr in [temp.T[4],temp.T[8]]:
        unique, counts = np.unique(arr, return_counts = True) 
        print('마스크: ',sum(counts[:6]),sum(counts[6:12]),sum(counts[12:18]))
        print('성별: ',sum(sum([counts[0:3],counts[6:9],counts[12:15]])),12600-sum(sum([counts[0:3],counts[6:9],counts[12:15]])))
        print(dict(zip(unique, counts)))
ans=[]
temp=temp.to_numpy()
for temp_list in temp:
    unique, counts = np.unique(temp_list, return_counts = True)
    uniq_cnt_dict = list(zip(unique, counts))
    uniq_cnt_dict=sorted(uniq_cnt_dict,key=lambda x : (x[1],x[0]),reverse=True)
    if len(uniq_cnt_dict)>1 :
        if len(unique)==2 and abs(uniq_cnt_dict[1][0]-uniq_cnt_dict[0][0])==1:
            if [uniq_cnt_dict[0][0],uniq_cnt_dict[1][0]]==[1,2]:
                if max(counts)-min(counts)<9:
                    print(dict(zip(unique, counts)))
                    cnt2+=2
                    ans.append(2)
                else:
                    ans.append(1)
            elif [uniq_cnt_dict[0][0],uniq_cnt_dict[1][0]]==[4,5]:
                if max(counts)-min(counts)<9:
                    print(dict(zip(unique, counts)))
                    cnt2+=2
                    ans.append(5)
                else:
                    ans.append(4)
            elif max(counts)-min(counts)<2:
                ans.append(uniq_cnt_dict[1][0])
            else:
                ans.append(uniq_cnt_dict[0][0])
        else:
            ans.append(uniq_cnt_dict[0][0])
    else:
        ans.append(uniq_cnt_dict[0][0])
info_path = os.path.join('/opt/ml/input/data/eval', 'info.csv')
info = pd.read_csv(info_path)
info['ans'] = ans
info.to_csv(os.path.join('./output', f'output_change_1to2_and_4to5.csv'), index=False)
