import torch.nn as nn
import torch.nn.functional as F
class lstm(nn.Module):
    def __init__(self,input_size=1,hidden_size=32,num_layers=1,output_size=30,dropout=0,batch_first=True):
        super(lstm,self).__init__()
        self.hidden_size=hidden_size
        self.input_size=input_size
        self.num_layers=num_layers
        self.output_size=output_size
        self.dropout=dropout
        self.batch_first=batch_first 
        self.rnn1=nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=3,batch_first=self.batch_first,dropout=self.dropout)
        # self.rnn2=nn.LSTM(input_size=self.hidden_size,hidden_size=32,num_layers=self.num_layers,batch_first=True,dropout=self.dropout)
        # self.linear1=nn.Linear(32,32)
        self.linear2=nn.Linear(32,30)
    def forward(self,x):
        # print("x",x.shape)
        o,(h,_)=self.rnn1(x)
        # print("layer1",o.shape)
        # print(h.shape)
        # hn=F.relu(hn)
        # out,(h,_)=self.rnn2(out)
        # print(out.shape)
        # print(h.shape)
        # hn=F.relu(hn)
        # print("layer2",hn.shape)
        # out=out.reshape()
        # print(c.shape)
        # out = self.linear1(out)
        # out=F.relu(out)
        # print("layer3",out.shape)
        out = self.linear2(h[-1])
        return out