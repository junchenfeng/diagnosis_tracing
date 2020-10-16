# -*- coding: utf-8 -*-

from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence
from torch import ne,where,eq,cat,no_grad

class Model(nn.Module):
    
    def __init__(self,num_items,hidden_size):
        
        super(Model,self).__init__()
        self._lstm  = nn.LSTM(num_items*2,hidden_size,num_layers=1,
                              batch_first = False,bidirectional=True)
        self._fc    = nn.Linear(hidden_size*2,2)
        self._ni    = num_items
        
    def rnn(self,items,actions,*args):
        
        mask    = ne(items,0).float().unsqueeze(2)
        lens    = ne(items,0).sum(1)
        inputs  = cat([F.one_hot(items,self._ni),F.one_hot(actions,self._ni)],dim=2).float()
        inputs  = inputs*mask
        
        pack_in = pack_padded_sequence(inputs.permute(1,0,2),lengths=lens,
                                       batch_first = False,enforce_sorted=False)
        pack_out,(h,c) = self._lstm(pack_in)
        out,_      = pad_packed_sequence(pack_out,batch_first=False)
        out        = out.permute(1,0,2)
        
        return h
        
    def _forward(self,items,actions,targets,training = True):
        """
            items and actions are 2-d tensor with shape [batch_size,sequence_length]
                with 0s as paddings in items and actions
                so 1 means wrong and 2 means right in actions
        """
        h = self.rnn(items,actions)
        h = F.dropout(h,training=training)
        feat = cat([h[0],h[1]],dim=-1)
#        feat   = states.sum(1)/ ne(items,0).sum(1,keepdim=True).float()      
        logits = self._fc(feat)
        
        if training:
            return F.cross_entropy(logits,targets)
        else:
            return logits
        
    def forward(self,items,actions,targets=None,training = True):
        if training:
            return self._forward(items,actions,targets,True)
        else:
            with no_grad():
                logits = self._forward(items,actions,targets,False)
                if targets is not None:
                    # accuracy
                    return (logits.argmax(1)==targets).float().mean().to('cpu').numpy()
                else:
                    # probabality of corrects
                    return F.softmax(logits,dim=1)[:,1].to('cpu').numpy()

if __name__ == '__main__':
    from torch import randint
    items   = randint(1,12,[13,17])
    actions = randint(0,2,[13,17])
    targets = randint(0,2,[13])
    items[10:,15:]   = 0
    actions[10:,15:] = 0
    model = Model(12,24)
    out   = model.forward(items,actions,targets,True)
        
        
        
        
    