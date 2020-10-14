from torch import tensor
from numpy.random import choice,shuffle

max_len = 128

def random_cut(length):
    s = choice(length-max_len+1)
    return s,s+max_len

def differeniate(statesA,statesB):
    return [[i,b] for i,(a,b) in enumerate(zip(map(int,statesA),map(int,statesB))) if b!=a]

def generate_targets(sequences):
    ciid    = -1
    Xs,Ys = [],[]
    for i,(iid,state) in enumerate(sequences):
        if iid != ciid:
            cstate = state # updata state
            ciid   = iid
        else:
            cstate = sequences[i-1][1]
        diff = differeniate(cstate,state)
        if len(diff) > 0:
            for d in diff:
                Xs.append(state)
                Ys.append(d)
    return Xs,Ys            
        
def pad_sequence3(sequences,padding = 0):
    """
        every element in sequence is a 2-d list
        with shape [time_steps,dim] 
        the dim is fixed
    """
    d2s = [len(seq) for seq in sequences]
    d3  = sequences[0][0].__len__()
    result = []
    max_l  = max(d2s)
    for seq,l in zip(sequences,d2s):
        result.append(seq + [[padding for _ in range(d3)] for i in range((max_l-l))])
    
    return result

def pad_sequence2(sequences,padding=0):
    lens = [len(seq) for seq in sequences]
    ml = max(lens)
    results = []
    for seq,l in zip(sequences,lens):
        results.append(seq+[padding for i in range(ml-l)])
    return results

def read_data(file_name):
    data = []
    add = data.append
    tmp  = []
    with open(file_name,'r',encoding='utf8') as f:
        while True:
            line = f.readline()
            if not line:break
            if line.strip()=="":
                if tmp!=[]:
                    add(tmp)
                tmp = []
            else:
                item,state = line.strip().split()
                # item + 1 bacause of padding value is 0
                tmp.append([int(item)+1,list(map(int,state))])
        if tmp != []:
            add(tmp)
    return data


def load_init():
    results = {}
    items   = {}
    for line in open('../data/items.dat','r',encoding='utf8'):
        itm,id_ = line.strip().split(' ')
        items[itm]=id_
    for line in open('../data/init.dat','r',encoding='utf8'):
        itm,state = line.strip().split(' ')
        results[int(items[itm])+1] = [1 if e=='0' else 0 for e in state]
    return results
                
class DataLoader():
    
    def __init__(self,data,inits):
        
        self.data  = data
        self.size  = len(data)  
        self.inits = inits
    
    def shuffle(self,):
        shuffle(self.data)
    
    def samples(self,batch_size):
        cursor = 0
        self.shuffle()
        while cursor < self.size:
            data = self.data[cursor:cursor+batch_size]
            cursor += batch_size
            states,masks = [],[]
            for d in data:
                if len(d)>max_len:
                    s,e = random_cut(len(d))
                    d = d[s:e]
                itms,sts = zip(*d)
                msk = [self.inits[i] for i in itms]
                states.append(list(sts))
                masks.append(msk)
            yield pad_sequence3(states),pad_sequence3(masks)
            
            
def check(Xs,Ys):
    for xs,ys in zip(Xs,Ys):
        for i in range(len(xs)-1):
            x_ = [v for v in xs[i]]
            if sum(x_) == 0:
                break
            pos,val = ys[i]
            x_[pos] = val
            if  x_ != xs[i+1]:
                print(x_)
                print('------')
                print(xs[i+1])
                print(' ')
                print(' ')
    print('ok')
        

           
if __name__ == '__main__':
    from numpy import array
    inits = load_init()
    if 'data' not in dir():
        data = read_data('../data/test.1.dat')
    dl = DataLoader(data,inits) 
    for x,y in dl.samples(100):
        x = array(x)
        y = array(y)
        print(x.shape,y.shape)
        break
#    
#    items,sequences = zip(*data[123])
#    x = data[15]
#    y = generate_targets(x)
#    items,states = zip(*x)
#    
#    x = [[1,[4,0,0,0]],
#         [1,[4,1,0,0]],
#         [1,[4,0,0,0]],
#         [1,[4,2,0,0]]]
#    
#    
#    targets = generate_targets(x)










    