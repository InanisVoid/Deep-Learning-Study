import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 

from torch.utils.data import DataLoader,Dataset 
from torchvision import transforms

class getData():
    def __init__(self,path,sequence_length=5,batchSize=16):
        self.data=pd.read_csv(path,usecols=["Open","Close","High","Low","Volume"])
        # print(self.data)
        # index_col="Date",parse_dates=["Date"],


        self.max=self.data['Close'].max()
        self.min=self.data['Close'].min()

        # volumeMean=self.data['Volume'].mean()
        # self.data['Volume']=self.data['Volume'].replace(0,volumeMean)
        self.data=self.data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))

        # print(self.data)

        self.sequence=sequence_length
        self.batchSize=batchSize

    def process(self):
        X=[]
        Y=[]
        # print(self.data['Close'])
        for i in range(self.data.shape[0]-self.sequence):
            #拆序列
            xdata=np.array(self.data.iloc[i:(i + self.sequence)].values, dtype=np.float32)
            # print(xdata)
            # print(xdata.shape)
            X.append(xdata)
            Y.append(np.array(self.data['Close'].iloc[i + self.sequence], dtype=np.float32))

        total_len=len(Y)
        trainX,trainY=X[:int(0.9*total_len)],Y[:int(0.9*total_len)]
        testX,testY=X[int(0.9*total_len):],Y[int(0.9*total_len):]

        train_loader = DataLoader(dataset=Mydataset(trainX, trainY), batch_size=self.batchSize, shuffle=True)
        test_loader = DataLoader(dataset=Mydataset(testX, testY), batch_size=self.batchSize, shuffle=False)

        # for i,data in enumerate(train_loader):
        #     print(i)
        #     x,y=data
        #     print(x,y)


        return self.max,self.min,train_loader,test_loader

    def plot(self):
        self.data.plot()
        plt.show()
        plt.close()



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
    Test=getData("ChinaBank.csv")
    minData,maxData,trainLoader,testLoader=Test.process()
    # Test.plot()
if __name__ == '__main__':
    main()