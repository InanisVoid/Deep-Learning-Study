import torch.nn as nn 

class sequentialTest(nn.Module):
    def __init__(self):
        super(sequentialTest,self).__init__()
        self.model = nn.Sequential(
            nn.LSTM(1,64,batch_first=True),
            # nn.LSTM(64,32),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,30)
        )
    def forward(self,x):
        out = self.model(x)
        return out
