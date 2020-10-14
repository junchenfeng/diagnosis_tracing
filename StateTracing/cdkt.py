# -*- coding: utf-8 -*-

from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence
from torch import where,ne,cat,no_grad,eq

num_items  = 5
hidden_size= 64
width   = 6
height  = 6
max_num = 7


class CDKT(nn.Module):
    
    def __init__(self,):
        super(CDKT,self).__init__()
        
        self.cnn = nn.Sequential(nn.Conv2d(max_num,hidden_size,3,2,padding = 1),
                                 nn.ELU(),
                                 nn.Conv2d(hidden_size,hidden_size,3,2,padding = 1),
                                 nn.ELU(),
                                 nn.Conv2d(hidden_size,hidden_size,2,1,0),
                                 nn.ELU())

        self.dcnn = nn.Sequential(nn.ConvTranspose2d(hidden_size,hidden_size,2,1),
                                  nn.ELU(),
                                  nn.ConvTranspose2d(hidden_size,hidden_size,3,2,1),
                                  nn.ELU(),
                                  nn.ConvTranspose2d(hidden_size,max_num,3,2,1,output_padding=1),
                                  nn.ELU())

        self._lstm = nn.LSTM(hidden_size,hidden_size,batch_first=False,bidirectional=False)

#        self._proj = nn.Sequential(nn.Linear(hidden_size,hidden_size,bias = False),
#                                   nn.Tanh())

    def encoding(self,sources,training):
        #the sources is a 3-d tensor with shape
        #       [batch_size,time_steps,width*height]
        #       though one-hot and reshape 
        #       the shape with be [batch_size,time_steps,7,width,height]
        lens = ne(sources.sum(2),0).sum(1)
        batch_size,time_steps,_ = sources.size()
        states  = F.one_hot(sources.reshape([batch_size*time_steps,width,height]),num_classes = max_num)
        cnns    = self.cnn(states.float().permute(0,3,1,2)).reshape([batch_size,time_steps,hidden_size])
        
        feat_in = F.dropout(cnns,p=0.5,training=training)
        
        pack_in = pack_padded_sequence(feat_in.permute(1,0,2),lens,batch_first=False,enforce_sorted=False )
        pack_out,_ = self._lstm(pack_in)
        pstates,_= pad_packed_sequence(pack_out,batch_first=False)
        
        return pstates.permute(1,0,2)
    
    def _forward(self,sources,masks,training=True):
        
        batch_size,time_steps,_ = sources.size()
        pstates    = self.encoding(sources[:,:-1,:],training)
        
        dec_states = self.dcnn(pstates.reshape([batch_size*(time_steps-1),hidden_size,1,1]))
        dec_states = dec_states.reshape([batch_size,time_steps-1,max_num,width*height]).permute(0,1,3,2)
 

        # where the student fill numbers 
        with no_grad():
            """in mask where 1s means student should fill those blanks"""
            rows,cols,pos = where(eq(masks[:,1:,:],1))
            targets = sources[:,1:,:][rows,cols,pos]
        logits  = dec_states[rows,cols,pos]
        
        # where the item's raw state
#        with no_grad():
#            """in mask where 0s means already given by items"""
#            rows_,cols_,pos_ = where(eq(masks[:,1:,:],0))
#            targets_ = sources[:,1:,:][rows_,cols_,pos_]
#        logits_  = dec_states[rows_,cols_,pos_]
        
        if training:
            return F.cross_entropy(logits,targets)
        else:
            acc = (logits.argmax(1) == targets).float().mean()
            return acc
        
    def forward(self,sources,masks,training):
        if training:
            return self._forward(sources,masks,True)
        else:
            with no_grad():
                return self._forward(sources,masks,False).to('cpu').numpy()
        

if __name__ == '__main__':
    from torch import randint
    model = CDKT()
    
    items   = randint(1,5,size=[13,17])
    sources = randint(0,7,size=[13,17,36])
    masks   = randint(0,2,size=[13,17,36])
    
    out     = model._forward(sources,masks,False)
        
        
        
        
        
            