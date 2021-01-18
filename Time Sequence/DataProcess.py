import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from fbprophet import Prophet

class datasetProcessor:
    def __init__(self):
        self.path=r"./Purchase Redemption Data/user_balance_table.csv"
        self.data=pd.read_csv(self.path,usecols=["report_date","total_purchase_amt","total_redeem_amt"],date_parser=["report_date"])
        # mm=MinMaxScaler()

        def p(x):
            x = list(str(x))
            # print(x)
            x.insert(6,"-")
            # print(x)
            x.insert(4,"-")
            # print(x)
            x="".join(x)
            return x

        self.data["report_date"]=self.data["report_date"].apply(p)

        self.purchase=self.data.groupby("report_date")["total_purchase_amt"].sum()
        self.redeem=self.data.groupby("report_date")["total_redeem_amt"].sum()


        self.purchase = pd.DataFrame({'ds':self.purchase.index,'y':self.purchase.values})
        self.redeem = pd.DataFrame({'ds':self.redeem.index,'y':self.redeem.values})

        self.purchase.to_csv("purchase.csv")
        self.redeem.to_csv("redeem.csv")



        self.redeem.plot(x="ds",y="y")
                    # self.data=pd.merge(self.purchase,self.redeem)
                    # self.data.to_csv(r"Data.csv")

                    # print(type(self.purchase))


                    # maxNum=self.purchase['y'].max()
                    # minNum=self.purchase['y'].min()
                    # self.purchase['y']=self.purchase['y'].apply(lambda x: (x - minNum) / (maxNum - minNum))
                    
                    
                    # self.purchase['y'] = mm.fit_transform(self.purchase['y'])

                    # print(self.purchase
                    # self.purchase=self.purchase.rename({'report_date':'ds','total_purchase_amt':'y'})
                    # self.redeem=self.redeem.rename(columns={'report_date':'ds','total_redeem_amt':'y'})
                    # print(self.data)
                    # self.purchase = mm.reverse_
                    # self.purchase["y"]=mm.fit_transform(self.purchase["y"])

                    # m=Prophet()
                    # m.fit(self.purchase)
                    # future = m.make_future_dataframe(periods=30)
                    # forecast=m.predict(future)
                    # m.plot(forecast)

                    # plt.plot("ds",y="y",label="True")
        plt.show()

def main():
    Test=datasetProcessor()

if __name__=="__main__":
    main()