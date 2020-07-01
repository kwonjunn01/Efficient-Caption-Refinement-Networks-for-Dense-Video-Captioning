import torch
import torch.nn as nn
import torch.nn.functional as F


class nAttention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.
    """

    def __init__(self, dim= 512):
        super(nAttention, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim*2, dim)
        self.linear2 = nn.Linear(dim, 1, bias=False)
        #self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, decoder_out):
        """
        Arguments:
            hidden{Variable} -- batch_size x dim, 64*512
            encoder_outputs {Variable} -- batch_size x seq_len x dim,64*100*512

        Returns:
            Variable -- context vector of size batch_size x dim
        """
        batch_size, seq_len, _ = decoder_out.size()
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        inputs = torch.cat((decoder_out, hidden),
                           2).view(-1, self.dim*2)
        o = self.linear2(F.tanh(self.linear1(inputs)))
        e = o.view(batch_size, seq_len)
        alpha = F.softmax(e, dim=1)
        
        context = torch.bmm(alpha.unsqueeze(1), decoder_out).squeeze(1)
        return context
