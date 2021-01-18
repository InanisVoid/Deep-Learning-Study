from ARIMA import ARIMA
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from pmdarima import auto_arima
from fbprophet import Prophet

class ArimaModel:
    def __init__(self,path):
        self.data=pd.read_csv(path)
        # maxNum=self.data['y'].max()
        # minNum=self.data['y'].min()
        
        
        # self.data['y']=self.data['y'].apply(lambda x: (x - minNum) / (maxNum - minNum))
    def process(self):  
        model = auto_arima(self.data['y'], max_P=10,max_D=2,max_Q=10,trace=True, error_action='ignore', suppress_warnings=True)
        model.fit(self.data['y'])
        forecast = model.predict(n_periods=30)


        # print(date)
        # df=pd.DataFrame({"ds":date,"forecast":forecast})
        # print(df)

        # plt.plot(df["ds"],df["forecast"])
        # plt.show()
        return forecast
        # print(forecast)

        # forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])



        # self.Arima=ARIMA(self.path,"y","2013-7","2014-8","2014-7","ds",False) 
        # self.Arima.plot()
        # self.Arima.diff()
        # self.Arima.acfBcf()

def processStr(x):
    x=str(x)
    x=x[0:10]
    # print(x)
    x=x.replace("-","")
    # print(x)
    return x

def main():

    f=pd.date_range(start='9/1/2014', periods=30)
    f=list(f.values)
    # print(f)
    f=list(map(processStr,f))
    # print(f)
    
    p=ArimaModel("purchase.csv")
    
    pForecast=p.process()
    r=ArimaModel("redeem.csv")
    rForecast=r.process()
    # # pForecast=pForecast[-30:]
    # pForecast=np.squeeze(pForecast)
    # # rForecast=rForecast[-30:]
    # rForecast=np.squeeze(rForecast.values)
    # print(f)
    print(pForecast)
    print(rForecast)

    ans=pd.DataFrame({'Date':f,'purchase':pForecast,'redeem':rForecast})

    print(ans)
    ans.to_csv("ArimaModel.csv",header=False,index=False)
if __name__ == '__main__':
    main()