# -*- coding: utf-8 -*-
from random import shuffle

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
            items,actions,_ = zip(*data)
            cursor += batch_size
            yield pad_sequence(items),pad_sequence(actions)
            
        


if __name__ == '__main__':
    data = read_data('../classic_kt.dat')
    dl   = DataLoader(data)
    for x,y in dl.sampling(72):
        break
            