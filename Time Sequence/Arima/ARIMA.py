import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns  

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt 
import statsmodels.tsa.arima_model as ArimaModel


class ARIMA():
    def __init__(self,path,col,begin,end,trainEnd,date="",Wash=True):
        #数据输入
        
        if(Wash):
            self.IN=pd.read_csv(path)
            self.IN=self.IN[self.IN[col].notna()]
            periods=pd.Periodself.INdex(year=self.IN["year"],month=self.IN["month"],day=self.IN["day"],hour=self.IN["hour"],freq="H")
            self.IN=self.IN.set_self.INdex(periods)
        else:
            self.IN=pd.read_csv(path,index_col = date, parse_dates=[date])

        inputs=self.IN[begin:end]
        self.data=inputs[col]
        self.Train=self.data[begin:trainEnd]
        self.TestBegin=trainEnd
    def plot(self):
        #可视化-原图
        self.data.plot()

    def diff(self):
        #差分
        self.differred=self.data.to_frame(name="Origin")
        self.differred["First"]=self.differred["Origin"].diff(1)
        self.differred["Second"]=self.differred["First"].diff(1)
        self.differred.plot(subplots=True,figsize=(18,12))

    def acfBcf(self):
        #ACF,BCF
        fig=plt.figure(figsize=(12,8))
        ax1=fig.add_subplot(211)
        fig=sm.graphics.tsa.plot_acf(self.data,lags=20,ax=ax1)#自相关
        ax1.xaxis.set_ticks_position('bottom')
        ax2=fig.add_subplot(212)
        fig=sm.graphics.tsa.plot_pacf(self.data,lags=20,ax=ax2)#偏自相关
        ax2.xaxis.set_ticks_position('bottom')
        fig.tight_layout()

    def train(self,p,d,q):
        #训练
        self.model = ArimaModel.ARIMA(self.Train,order=(p,d,q)).fit()

    def predict(self,dynamic=False):
        #预测
        predicted = self.model.predict(start=self.TestBegin,dynamic=dynamic,typ='levels')
        _, ax = plt.subplots(figsize=(12, 8))
        ax = self.Train.plot(ax=ax)
        predicted.plot(ax=ax)

    def __del__(self):
        plt.show()
        plt.close()



def main():
    arima=ARIMA('./ARIMA/ChinaBank.csv','Close','2014-1','2014-6','2014-4','Date',False)
    arima.train(1,0,0)
    arima.predict()
    # arima.acfBcf()

if __name__ == '__main__':
    main()