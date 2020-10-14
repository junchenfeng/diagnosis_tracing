# -*- coding: utf-8 -*-

from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence
from torch import ne,where,eq,cat,no_grad

class Model(nn.Module):
    
    def __init__(self,num_items,hidden_size):
        
        super(Model,self).__init__()
        self._lstm  = nn.LSTM(num_items*2,hidden_size,num_layers=1,
                              batch_first = False,bidirectional=False)
        self._ni      = num_items
        self._trans   = nn.Embedding(num_items,hidden_size,padding_idx=0)
        self._querier = nn.Linear(hidden_size,2)
        
    def rnn(self,items,actions):
        
        mask    = ne(items,0).float().unsqueeze(2)
        lens    = ne(items,0).sum(1)
        inputs  = cat([F.one_hot(items,self._ni),F.one_hot(actions,self._ni)],dim=2).float()
        inputs  = inputs*mask
        
        pack_in = pack_padded_sequence(inputs.permute(1,0,2),lengths=lens,
                                       batch_first = False,enforce_sorted=False)
        pack_out,_ = self._lstm(pack_in)
        out,_      = pad_packed_sequence(pack_out,batch_first=False)
        out        = out.permute(1,0,2)
        
        return out
        
    def _forward(self,items,actions,training = True):
        """
            items and actions are 2-d tensor with shape [batch_size,sequence_length]
                with 0s as paddings in items and actions
                so 1 means wrong and 2 means right in actions
        """
        states = self.rnn(items[:,:-1],actions[:,:-1])
        query  = self._querier(self._trans(items[:,1:]) * states)
        
        states = F.dropout(states,training=training)
        query  = F.dropout(query ,training=training)
        
        """find masks"""
        rows,cols = where(ne(items[:,1:],0))
        logits    = query[rows,cols]
        targets   = actions[:,1:][rows,cols]
        
        if training:
            return F.cross_entropy(logits,targets)
        else:
            return logits,targets
        
    def forward(self,items,actions,training = True):
        if training:
            return self._forward(items,actions,True)
        else:
            with no_grad():
                logits,targets = self._forward(items,actions,False)
                return logits.to('cpu').numpy(),targets.to('cpu').numpy()
    

if __name__ == '__main__':
    from torch import randint
    items   = randint(1,12,[13,17])
    actions = randint(0,2,[13,17])
    items[10:,15:]   = 0
    actions[10:,15:] = 0
    model = Model(12,24)
    out   = model._forward(items,actions,False)
        
        
        
        
    