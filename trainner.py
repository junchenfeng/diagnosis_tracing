import csv
from   rnn import *
from torch import optim
from torch import tensor
from torch.nn.utils import clip_grad_value_
from torch.nn import CrossEntropyLoss

def load_data(file_name,tid = None):
    fo = open(file_name,'r',encoding='utf8')
    reader = csv.reader(fo)
    data = []
    i=1
    while True:
        try:
            line = next(reader)
            usr = line[0]
            items = next(reader)
            actions = next(reader)
            targets = next(reader)
            actions = [int(act) for act in actions if act.strip()]
            items   = [int(itm) for itm in items if itm.strip()]
            targets = [int(tgt) for tgt in targets]
            if tid is not None:
                targets = targets[tid]
            if actions.__len__()==0:
                continue
            data.append([items,actions,targets])
            i+=1
        except StopIteration:
            break
    fo.close()
    return data

if __name__ == '__main__':
    
    data  = load_data('data.csv',0)
    dl    = DataLoader(data)
    model = Model(12,128) 
    model = model.cuda()
    loss_func = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),5e-4)
    
    for i in range(1000):
        
        x,y    = dl.sampling(72)
        inps   = tensor(x).long()
        logits = model.forward(inps.cuda())
        loss   = loss_func(logits,tensor(y).cuda())
        
        optimizer.zero_grad()
        clip_grad_value_(model.parameters(),10)
        loss.backward()
        optimizer.step()
        
        if i%100 == 0:
            print(i,loss.data.cpu().numpy())
        
        
        
        



