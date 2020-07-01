import torch
import torch.nn as nn
from .nAttention import nAttention

class nEncoderRNN(nn.Module):
    def __init__(self, input_dropout_p=0.2, rnn_dropout_p=0.5, bidirectional=False):
        super(nEncoderRNN, self).__init__()
        self.rnn_cell = nn.GRU
        self.rnn_dropout_p = rnn_dropout_p
        self.rnn = self.rnn_cell(1024, 512, 1,bidirectional= bidirectional, batch_first=True, dropout=self.rnn_dropout_p)
        self.input_dropout_p = input_dropout_p
        
        self.lin = nn.Linear(1, 24)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.nattention = nAttention(512)
        self.max_len = 100
        self.linear = nn.Linear(6831, 512)
        self.gongal = torch.zeros(64, 1, 512)
        self.dropout = nn.Dropout(input_dropout_p)



    def _init_hidden(self):
        return nn.init.xavier_normal_(self.lin.weight).unsqueeze(0).repeat(64, 1, 1)

    def forward(self, hidden, decoder_out):
        decoder_out = self.linear(decoder_out)
        _, lenn, __ = decoder_out.size()
        context = self.nattention(hidden.squeeze(0), decoder_out)
        for i in range(lenn - 1):
            encoder_input = torch.cat([decoder_out[:,i,:], context], dim=1)
            encoder_input = self.dropout(encoder_input).unsqueeze(1)
            _, hidden = self.rnn(encoder_input, hidden)
        return hidden
