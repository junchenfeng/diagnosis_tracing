# -*- coding: utf-8 -*-
import json
from random import shuffle
from numpy.random import choice as nchoice




def pad_sequence(sequences,padding_value=0):
    lens = [len(seq) for seq in sequences]
    results = []
    ml   = max(lens)
    for seq,l in zip(sequences,lens):
        results.append(seq+[padding_value for i in range(ml-l)])
    return results

def read_data(file_name,targets=['11612','11613'],keep=0):
    data  = []
    item2id = {}
    with open(file_name,'r',encoding='utf8') as f:
        while True:
            items   = f.readline()
            actions = f.readline()
            if (not items) or (not actions):
                break
            items   = items.strip().split(',')
            for itm in items:
                if itm not in item2id:
                    item2id[itm] = len(item2id)+1
            item_ids = [item2id[itm] for itm in items]
            actions  = list(map(int,actions.strip().split(',')))
            Is,As,Ts = [],[],[list() for t in targets]
            for itm,iid,act in zip(items,item_ids,actions):
                try:
                    idx = targets.index(itm)
                    Ts[idx].append(act)
                except:
                    Is.append(iid)
                    As.append(act)
            if len(Is)<3:
                continue
            if Ts[keep] == []:
                continue
            
            data.append([Is,As,Ts[keep][0]])
    f = open('item2id.tmp','w',encoding='utf8')
    for itm,id_ in item2id.items():
        f.write(f'{itm} {id_}\n')
    f.close()
    
    
    
    return data

#class DataLoader():
#    def __init__(self,data):
#        self.data = {}
#        for x,y,t in data:
#            if t not in self.data:
#                self.data[t] = []
#            self.data[t].append([x,y])

class DataLoader():
    """
        classic DKT dataloader
    """
    def __init__(self,data):
        
        self.data = data
    
    def sampling(self,batch_size):
        shuffle(self.data)
        cursor = 0
        while cursor < self.data.__len__():
            data =self.data[cursor:cursor+batch_size]
            items,actions,targets = zip(*data)
            cursor += batch_size
            yield pad_sequence(items),pad_sequence(actions),targets

class DataLoader_balanced():
    """
        dataloader for anomaly detection
    """
    def __init__(self,data):
        
        self.data = {}
        for x,y,t in data:
            if t not in self.data:
                self.data[t] = []
            self.data[t].append([x,y])
        self.keys = list(self.data.keys())
        self.lens = {k:len(value) for k,value in self.data.items()}
    
    def sampling(self,batch_size):
        
        items,actions,targets = [],[],[]
        for i in range(batch_size):
            t = nchoice(self.keys)
            p = nchoice(self.lens[t])
            itm,act = self.data[t][p]
            items.append(itm)
            actions.append(act)
            targets.append(t)
        return pad_sequence(items),pad_sequence(actions),targets
    
    def export(self,batch_size=72) :
        X,Y=[],[]
        for t in self.data:
            for d in self.data[t]:
                X.append(d)
                Y.append(t)
        cursor = 0
        while cursor < len(X):
            xs      = X[cursor:cursor+batch_size]
            targets = Y[cursor:cursor+batch_size]
            cursor += batch_size
            items,actions = list(zip(*xs))
            yield pad_sequence(items),pad_sequence(actions),targets
            
            

    
if __name__ == '__main__':
    data = read_data('../data/classic_kt.dat')
    dl   = DataLoader_balanced(data)
    for x,y,z in dl.export(10):
        break