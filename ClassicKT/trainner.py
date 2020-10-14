# -*- coding: utf-8 -*-
from torch import tensor
from torch import cuda
from torch import optim
from torch.nn.utils import clip_grad_value_
from model import Model
from numpy import argmax,concatenate
from dataloader import read_data,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,accuracy_score

cuda.empty_cache()



data = read_data('../classic_kt.dat')
train_data,test_data = train_test_split(data,test_size=.2)

model = Model(13,64)
model = model.cuda()
optimizer = optim.Adam(model.parameters(),5e-4)

dl_train = DataLoader(train_data)
dl_test  = DataLoader(test_data)
for r in range(10): # 10-epochs
    i = -1
    print('training:')
    for x,y in dl_train.sampling(72):
        i+=1
        loss = model.forward(tensor(x).cuda(),tensor(y).cuda(),True)
        optimizer.zero_grad()
        clip_grad_value_(model.parameters(),10)
        loss.backward()
        optimizer.step()
        
        if i%100 == 0:
            loss_val        = loss.data.to('cpu').numpy().tolist()
            logits,targets  = model.forward(tensor(x).cuda(),tensor(y).cuda(),False)
            acc_val = (argmax(logits,axis=1) == targets).mean() 
            print(f'    {r:<5d}--{i:<5d}--{loss_val:.3f}--{acc_val:.3f}%')
    
    y_prob,y_pred,y_true = [],[],[]
    for x,y in dl_test.sampling(100) :
        logits,targets = model.forward(tensor(x).cuda(),tensor(y).cuda(),False)
        y_prob.append(logits[:,1])
        y_pred.append(argmax(logits,1))
        y_true.append(targets)
    y_prob  = concatenate(y_prob,0)
    y_true  = concatenate(y_true,0)
    y_pred  = concatenate(y_pred,0)
    
    fpr,tpr,thres = roc_curve(y_true,y_prob)
    auc_score     = auc(fpr,tpr)
    acc_score     = accuracy_score(y_true,y_pred)
    
    print('testing')
    print(f'    {r:<5d}--acc:{acc_score:.3f}%----auc:{auc_score:.3f}')
    

    
    
    