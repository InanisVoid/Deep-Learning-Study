import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 

from torch.utils.data import DataLoader,Dataset 
from torchvision import transforms

class getData():
    def __init__(self,path,sequence_length=40,predict_length=30,batchSize=16):
        self.data=pd.read_csv(path,usecols=["y"])
        self.max=self.data['y'].max()
        self.min=self.data['y'].min()
        self.data=self.data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
        self.data.to_csv("Test.csv")
        self.predictLength=predict_length
        self.sequence=sequence_length
        self.batchSize=batchSize

    def process(self):
        X=[]
        Y=[]
 
        for i in range(self.data.shape[0]-self.predictLength-self.sequence+1):
            #拆序列
            xdata=np.array(self.data.iloc[i:(i + self.sequence)].values, dtype=np.float32)
            # print()
            X.append(xdata)
            Y.append(np.array(self.data.iloc[(i + self.sequence):(i + self.sequence+self.predictLength)].values, dtype=np.float32))

        # print(X)
        # print(Y)

        total_len=len(Y)

        trainX,trainY=X[:-1],Y[:-1]
        testX,testY=X[-1:],Y[-1:]
        # fT=lambda x: x*(self.max - self.min) + self.min
        
        # print(list(map(fT,testY)))


        train_loader = DataLoader(dataset=Mydataset(trainX, trainY), batch_size=self.batchSize, shuffle=False)
        test_loader = DataLoader(dataset=Mydataset(testX, testY), batch_size=self.batchSize, shuffle=False)


        return self.max,self.min,train_loader,test_loader



class Mydataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.tranform != None:
            return self.tranform(x1), y1
        return x1, y1

    def __len__(self):
        return len(self.x)

def main():
    Test=getData("purchase.csv")
    minData,maxData,trainLoader,testLoader=Test.process()
    # Test.plot()
if __name__ == '__main__':
    main()