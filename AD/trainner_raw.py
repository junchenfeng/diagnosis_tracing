# -*- coding: utf-8 -*-
from torch import tensor
from torch import cuda
from torch import optim
from torch.nn.utils import clip_grad_value_
from model import Model
from dataloader import read_data,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc,roc_curve

cuda.empty_cache()

def compute_auc(y_true,y_prob):
    fpr,tpr,thres = roc_curve(y_true,y_prob)
    auc_score     = auc(fpr,tpr)
    return auc_score



data = read_data('../data/classic_kt.dat')
train_data,test_data = train_test_split(data,test_size=.2)

model = Model(13,64)
model = model.cuda()
optimizer = optim.Adam(model.parameters(),5e-4)

dl_train = DataLoader(train_data)
dl_test  = DataLoader(test_data)

for ep in range(10): # 10-epochs
    i = 0
    for x,y,z in dl_train.sampling(72):
        loss = model.forward(tensor(x).cuda(),tensor(y).cuda(),
                             tensor(z).long().cuda(),True)
        optimizer.zero_grad()
        clip_grad_value_(model.parameters(),10)
        loss.backward()
        optimizer.step()
        i+=1
    
        if i%100 == 0:
            loss_val = loss.data.to('cpu').numpy().tolist()
            acc_val  = model.forward(tensor(x).cuda(),tensor(y).cuda(),
                                     tensor(z).long().cuda(),False)
            print(f'training :{i:<5d}--{loss_val:.3f}--{acc_val:.3f}%')

    loss_val = loss.data.to('cpu').numpy().tolist()
    acc_val  = model.forward(tensor(x).cuda(),tensor(y).cuda(),
                             tensor(z).long().cuda(),False)
    print(f'training :{i:<5d}--{loss_val:.3f}--{acc_val:.3f}%')
    
    accs   = [] 
    y_prob = []
    y_true = []
    for x,y,z in dl_test.sampling(100) :
        acc   = model.forward(tensor(x).cuda(),tensor(y).cuda(),
                              tensor(z).long().cuda(),False)
        probs = model.forward(tensor(x).cuda(),tensor(y).cuda(),None,False)
        accs.append(acc)
        y_prob.extend(probs.tolist())
        y_true.extend(z)
    total_acc = sum(accs)/len(accs)
    auc_score = compute_auc(y_true,y_prob)
    print(f'testing  :{i:<5d}--{total_acc:.3f}%--{auc_score:.3f}')
