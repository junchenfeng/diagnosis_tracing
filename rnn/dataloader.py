from torch import tensor
from numpy.random import choice,shuffle



def pad_sequence(sequences,padding = 0):
    lens = [len(seq) for seq in sequences]
    result = []
    max_l  = max(lens)
    for seq,l in zip(sequences,lens):
        result.append(seq+[padding]*(max_l-l))
    return result


class DataLoader():
    
    def __init__(self,raw_data,test_size=.2):
        
        data = {}
        for items,actions,target in raw_data:
            if target not in data:
                data[target]=[]
            data[target].append([items,actions])
        self.labels = list(data.keys())
        
        self.train_data = {}
        self.test_data  = {}
        for k in data:
            shuffle(data[k])
            l = len(data[k])
            if l < 10:
                raise Exception('not enough data')
            n = int(l*test_size)
            self.test_data[k] = data[k][:n]
            self.train_data[k] = data[k][n:]

    def sampling(self,batch_size):
        X,Y = [],[]
        targets = choice(self.labels,size=batch_size).tolist()
        for t in targets:
            l = len(self.train_data[t])
            sample_id = choice(l)
            items,actions = self.train_data[t][sample_id]
            X.append([(i+1)*2+a for i,a in zip(items,actions)])
            Y.append(t)
        X = pad_sequence(X)
        return X,Y
            
            
        