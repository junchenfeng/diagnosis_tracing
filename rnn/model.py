from torch import nn
from torch import ne
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.nn.functional import one_hot



class Model(nn.Module):
    """
        item id starts from 1 'cause 0s are paddings
    """
    
    def __init__(self,num_items,hidden_size):
        
        super(Model,self).__init__()
        input_size  = (num_items+1)*2
        hidden_size = hidden_size //2 * 2
        self._lstm  = nn.LSTM(input_size,hidden_size//2,num_layers=1,
                              bidirectional=True,batch_first=False)
        self._fc    = nn.Linear(hidden_size,2)
        self._ni    = num_items + 1
    
    def forward(self,input_ids):
        # convert labels to one-hot code
        inputs = one_hot(input_ids,self._ni * 2).float()
        # through the rnn layer
        seq_lengths = ne(input_ids,0).sum(1)
        inputs_ = inputs.permute(1,0,2)
        packed_sequence = pack_padded_sequence(inputs_,seq_lengths,
                                               batch_first=False,
                                               enforce_sorted = False)
        packed_out,_    = self._lstm(packed_sequence,)
        outputs, _      = pad_packed_sequence(packed_out, 
                                              batch_first = False)
        rnn_out = outputs.permute(1,0,2)
        rep = rnn_out.max(1).values
        logits = self._fc(rep)
        
        return logits
    
if __name__ == '__main__':
    from torch.nn.utils.rnn import pad_sequence
    from torch import tensor
    inp   = tensor([[1,2,3,0],[4,5,6,2]])
    act   = tensor([[0,0,1,0],[1,1,1,0]])
    model = Model(7,20)
    out   = model.forward(inp)
    