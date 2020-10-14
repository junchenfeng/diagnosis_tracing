# -*- coding: utf-8 -*-
from torch import nn
from torch.nn import functional as F
from torch import ne,cat,where,rand,lt,argmax,no_grad
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence


hidden_size = 64
width   = 6
height  = 6
max_num = 7

class S2S(nn.Module):
    def __init__(self):
        super(S2S,self).__init__()
#        self._table = nn.Embedding(num_items,hs,padding_idx=0)
        
        self.cnn = nn.Sequential(nn.Conv2d(max_num,hidden_size,3,2,padding = 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(hidden_size,hidden_size,3,2,padding = 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(hidden_size,hidden_size,2,1,0),
                                 nn.ReLU(inplace=True))

        self.dcnn = nn.Sequential(nn.ConvTranspose2d(hidden_size,hidden_size,2,1),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(hidden_size,hidden_size,3,2,1),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(hidden_size,max_num,3,2,1,output_padding=1),
                                  nn.ReLU())
        
        self.encoder = nn.LSTM(hidden_size,hidden_size//2,bidirectional=True,
                               batch_first=False)
        
        self.decoder = nn.LSTM(hidden_size,hidden_size,
                               bidirectional=False,batch_first=False)
        
        # the initial decode state
        self.init_dstate = nn.Parameter(rand(1,1,hidden_size),requires_grad=True)
        
    
    def _encoding(self,sources):
        #the sources is a 3-d tensor with shape
        #       [batch_size,time_steps,width*height]
        #       though one-hot and reshape 
        #       the shape with be [batch_size,time_steps,7,width,height]
        batch_size,time_steps,_ = sources.size()
        
        states  = F.one_hot(sources).reshape([batch_size*time_steps,width,height,max_num]).float()
        cnns    = self.cnn(states.permute(0,3,1,2)).reshape([batch_size,time_steps,hidden_size])
#        embs    = self._table(items)
#        feat_in = cat([cnns,embs],dim=2)
        feat_in = cnns
        lens    = ne(sources.sum(2),0).sum(1).long()
        pack_in = pack_padded_sequence(feat_in.permute(1,0,2),lens,batch_first=False,enforce_sorted=False )
        pack_out,(h,c) = self.encoder(pack_in)
        en_states,_    = pad_packed_sequence(pack_out,batch_first=False)
        if self.encoder.bidirectional:
            h = cat([h[:1],h[1:]],dim=2)
            c = cat([c[:1],c[1:]],dim=2)
        
        return en_states,(h,c)  

    
    def _decoding(self,enc_states,hidden,targets,teach_force=.5):
        """
            enc_states is used for attention decoding
            args:
                targets : in shape [batch_size,time_steps,dim]
                          dim is width * height
        """
        batch_size,max_decode_steps,_ = targets.size()
        device = targets.device
        # init inputs
        feat_in = self.init_dstate.repeat([1,batch_size,1])
        decode_states = []
        for i in range(max_decode_steps):
            dec_out,hidden = self.decoder(feat_in,hidden)
            rec_out = self.dcnn(dec_out.reshape([batch_size,hidden_size,1,1])).permute(0,2,3,1).reshape([batch_size,1,width*height,max_num])
            decode_states.append(rec_out)
            # teacher force
            # for next step
            if teach_force and  (0 < teach_force < 1):
                inp   = F.one_hot(targets[:,i,:]).reshape([batch_size,width,height,max_num]).permute(0,3,1,2)
                guide = self.cnn(inp.float()).reshape([1,batch_size,hidden_size])
                # teach-force random select index
                index    = rand([1,batch_size,1]).repeat([1,1,hidden_size]).to(device)
                feat_in  = where(lt(index,teach_force),dec_out,guide)
            else:
                feat_in  = dec_out 
        return cat(decode_states,dim=1)
    
    
    def _comput_loss(self,dec_out,targets):
        """
            compute loss while mask paddings
        """
        rows,cols     = where(ne(targets.sum(2),0))
        valid_targets = targets[rows,cols]
        valid_logits  = dec_out[rows,cols]
        
        return F.cross_entropy(valid_logits.reshape([-1,max_num]),
                               valid_targets.flatten())
                
    def forward(self,sources,targets,teach_force=.5):
        #the input is a 3-d tensor with shape
        #    [batch_size,time_steps,width*height]
        #    though one-hot and reshape 
        #    the shape with be [batch_size,time_steps,10,width,height]
        enc_states,(h,c) = self._encoding(sources)
        h = F.dropout(h,training=True,inplace = True)
        c = F.dropout(c,training=True,inplace = True)
        dec_out = self._decoding(enc_states,(h,c),targets,teach_force)
        
        return self._comput_loss(dec_out,targets)
    
    
    def predicts(self,sources,max_len):
        enc_states,(h,c) = self._encoding(sources)
        outputs = []
        # init state
        batch_size = sources.size(0)
        device     = sources.device
        with no_grad():
            feat_in = self.init_dstate.repeat([1,batch_size,1]).to(device)
            for s in range(max_len):
                feat_in,(h,c) = self.decoder(feat_in,(h,c))
                rec_out       = self.dcnn(feat_in.reshape([batch_size,hidden_size,1,1])).permute(0,2,3,1).reshape([batch_size,1,width*height,max_num])
                outputs.append(rec_out.argmax(-1))
        return cat(outputs,dim=1)
    
    
    def compute_accuracy(self,preds,targets):
        with no_grad():
            rows,cols = where(ne(targets.sum(2),0))
        return (preds[rows,cols] == targets[rows,cols]).float().mean().data.to('cpu').numpy()
            
        
        


if __name__ == '__main__':
    from torch import rand,randint
    model = S2S()
#    items  = randint(0,20,size=[13,17])
    sources = randint(1,7,size=[13,17,36])
    targets = randint(1,7,size=[13,23,36])
    targets[10:,20:]=0
    outputs = model.predicts(sources,23)
    acc     = model.compute_accuracy(outputs,targets)
    
#    x = randint(0,10,size=[13,17,36])
#    o  = F.one_hot(x)
#    o_ = o.reshape([13,17,6,6,10]) 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    