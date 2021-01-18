import torch
import torch.nn as nn 
from LSTM import lstm 
from DataSet import getData
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np 
import matplotlib.pyplot as plt 

class lstmTrain():
    def __init__(self,path,device,rate,f=nn.MSELoss()):
        self.Data=getData(path)
        self.minData,self.maxData,self.trainLoader,self.testLoader=self.Data.process()
        self.device=device
        self.model=lstm().to(device)
        self.criterion=f
        self.rate=rate
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.rate)
        self.writer=SummaryWriter('./log')

    def train(self,epochs):
        for i in range(epochs): 
            total_loss=0
            for _,(inputs,labels) in enumerate(self.trainLoader):
                inputs,labels=inputs.to(self.device),labels.to(self.device) #GPU
                
                # print(inputs)
                # print(inputs.shape)

                pred = self.model(inputs)
                pred=pred.squeeze()

                # print(pred)
                # print(pred.shape)
                
                labels=labels.squeeze()
                # print(labels)
                # print(labels.shape)
                
                loss = self.criterion(pred, labels)


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.writer.add_scalar('Train/Loss',loss.item(),i)
            print(i,total_loss)
        torch.save(self.model,"Model.pkl")

    def load(self,path):
        self.model=torch.load(path)

    def predict(self):
        self.prediction=[]
        self.truth=[]
        for _,(inputs,labels) in enumerate(self.testLoader):
                inputs,labels=inputs.to(self.device),labels.to(self.device) #GPU
                print(inputs)
                pred = self.model(inputs)
                print (labels)
                pred=pred.squeeze()

                self.prediction+=pred.tolist()
                self.truth+=labels.tolist()                
        
        # print(prediction,truth)
        
    def predict_plot(self):

        # print(self.truth.shape)
        # print(self.prediction)
        # self.prediction=self.prediction[0]
        # print(self.truth)
        self.truth=self.truth[0]

        # print(self.prediction)
        # f=lambda x: x*(self.maxData - self.minData) + self.minData
        # fT=lambda x: x[0]*(self.maxData - self.minData) + self.minData
        f = lambda x:x
        fT = lambda x:x[0]

        self.prediction=list(map(f,self.prediction))
        self.truth=list(map(fT,self.truth))
        print(self.truth)
        

        xlen=len(self.prediction)
        xlabel=np.arange(0,xlen)
        # print(self.truth)



        for x,p,t in zip(xlabel,self.prediction,self.truth):
            # print(p)
            # print(t)
            # print(x)
            self.writer.add_scalars('Test1',{"Prediction":p,"True Data":t},x)

        plt.plot(xlabel,self.prediction,label="Prediction") 
        plt.plot(xlabel,self.truth,label="True Data")
        plt.show()

    # def loss_plot():
        




def main():
    rate=0.0001
    device=torch.device("cuda:0")
    Test=lstmTrain("purchase.csv",device,rate)
    Test.train(500)
    # Test.load("Model.pkl")
    Test.predict()
    Test.predict_plot()
if __name__ == '__main__':
    main()