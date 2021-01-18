from fbprophet import Prophet
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

class prophetModel():
    def __init__(self,path):
        self.data=pd.read_csv(path)
    def process(self):
        maxNum=self.data['y'].max()
        minNum=self.data['y'].min()

        
        self.data['y']=self.data['y'].apply(lambda x: (x - minNum) / (maxNum - minNum))

        m=Prophet()
        m.fit(self.data)
        future = m.make_future_dataframe(periods=30)
        forecast=m.predict(future)
        m.plot(forecast)
        forecast=forecast["yhat"]

        forecast=forecast.apply(lambda x: x*(maxNum - minNum)+minNum)
        # print
        return future,forecast
        # plt.show()    

def processStr(x):
    x=str(x)
    x=x[0:10]
    # print(x)
    x=x.replace("-","")
    # print(x)
    return x
def main():
    p=prophetModel("purchase.csv")
    f,pForecast=p.process()
    r=prophetModel("redeem.csv")
    _,rForecast=r.process()
    f=f[-30:]
    f["ds"]=f["ds"].apply(processStr)
    f=np.squeeze(f.values)
    pForecast=pForecast[-30:]
    pForecast=np.squeeze(pForecast.values)
    rForecast=rForecast[-30:]
    rForecast=np.squeeze(rForecast.values)
    print(f)
    print(pForecast)
    print(rForecast)

    ans=pd.DataFrame({'Date':f,'purchase':pForecast,'redeem':rForecast})

    print(ans)
    ans.to_csv("ANS2.csv",header=False,index=False)

if __name__=="__main__":
    main()