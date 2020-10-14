# -*- coding: utf-8 -*-
from torch import optim
from torch import tensor,save
from torch import cuda
from torch.nn.utils import clip_grad_value_

from dataloader import read_data,DataLoader,load_init
from cdkt import CDKT


use_cuda = True

if use_cuda:
    cuda.empty_cache()

""" training mode"""
results = []
f = 3

model = CDKT()
if use_cuda:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(),5*1e-4)
DL = DataLoader(read_data(f'/data/train.{f}.dat'),load_init())
for r in range(10): # 20-epochs
    i = 0
    for x,y in DL.samples(72):
        X = tensor(x)
        Y = tensor(y)
        if use_cuda:
            X = X.cuda()
            Y = Y.cuda()
        loss = model.forward(X,Y,True)
        
        optimizer.zero_grad()
        clip_grad_value_(model.parameters(),10)
        loss.backward()
        optimizer.step()
        
        i += 1
        if i%100 == 0:
            loss_val = loss.data.to('cpu').numpy()
            print(f'{r:5d}--{i:5d}--{loss_val:.3f}')
    
    loss_val = loss.data.to('cpu').numpy()
    print(f'{r:5d}--{i:5d}--{loss_val:.3f}')       

"""on testing """
results = []
DL = DataLoader(read_data(f'/data/test.{f}.dat'),load_init())
for x,y in DL.samples(100):
    X     = tensor(x)
    Y     = tensor(y)
    if use_cuda:
        X = X.cuda()
        Y = Y.cuda()
    acc = model.forward(X,Y,False)
    results.append(acc.tolist())

total_acc = sum(results) / len(results)
print(total_acc) 





















