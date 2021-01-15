import torch.nn as nn
class lstm(nn.Module):
    def __init__(self,input_size=5,hidden_size=32,num_layers=1,output_size=1,dropout=0,batch_first=True):
        super(lstm,self).__init__()
        self.hidden_size=hidden_size
        self.input_size=input_size
        self.num_layers=num_layers
        self.output_size=output_size
        self.dropout=dropout
        self.batch_first=batch_first 
        self.rnn=nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=self.batch_first,dropout=self.dropout)
        self.linear=nn.Linear(self.hidden_size,self.output_size)
    def forward(self,x):
        _,(hidden,_)=self.rnn(x)
        out = self.linear(hidden)
        return out