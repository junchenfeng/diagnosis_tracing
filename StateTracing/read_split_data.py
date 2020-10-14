# -*- coding: utf-8 -*-

import os
from sklearn.model_selection import KFold



#def get_init(file_name):
#    
#    result = {}
#    for line in open(file_name,'r',encoding='utf8'):
#        itm,init_state = line.strip().split()
##        init_state = list(map(int,init_state))
#        result[itm] = init_state
#    return result

#def process_states(sequence,init_states):
#    """ add init states and make diff"""
#    results = []
#    for i,(item,state) in enumerate(sequence):
#        results.append([init_states[item],diff(init_states[item],state)])
#    return results

def load_data(file_name,headers=True):
    total_data = []
    items = {}
    add = total_data.append
    with open(file_name,'r',encoding='utf8') as f:
        if headers:
            line = f.readline()
        cusr = None
        tmp  = []
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            usr,item,ctime,state = line.strip().split('\t')
            if item not in items:
                items[item] = str(len(items))
            iid = items[item]
            state = state.replace(',','')
            if usr != cusr:
                if tmp!=[]:
                    #process here
                    tmp.sort(key=lambda x:x[0])
                    add([[itm,state] for _,itm,state in tmp])
                cusr = usr
                tmp  = [[float(ctime),iid,state]]
            else:
                tmp.append([float(ctime),iid,state])
        if tmp!=[]:
            tmp.sort(key=lambda x:x[0])
            add([[i,s] for _,i,s in tmp])
    return items,total_data  


def kfold_split(data,f=8):
    kfold = KFold(n_splits=f,shuffle=True)
    if not os.path.exists('data'):
        os.mkdir('data')
    for k,(train_idxs,test_idxs) in enumerate(kfold.split(data)):
        with open(f'data/train.{k+1}.dat','w',encoding='utf8') as f:
            for idx in train_idxs:
                for itm,state in data[idx]:
                    f.write(f'{itm} {state}\n')
                f.write('\n')
        with open(f'data/test.{k+1}.dat','w',encoding='utf8') as f:
            for idx in test_idxs:
                for itm,state in data[idx]:
                    f.write(f'{itm} {state}\n')
                f.write('\n')
    return 



def main(file_name,f=8):
    items,data  = load_data(file_name)
    kfold_split(data,f)
    with open('data/items.dat','w',encoding='utf8') as f:
        for itm,id_ in items.items():
            f.write(f'{itm} {id_}\n')
    return True
    
          
            
        
        

if __name__ == '__main__':
    file_name = '../2020-02-10_2020-02-23.csv'
    out = main(file_name)
#    if 'data' not in dir():
#        data = load_data('../2020-02-10_2020-02-23.csv') 
    
#    data_ = [process_states(seq,init_states) for seq in data]
#    out  = process_states(data[1234],init_states)          












