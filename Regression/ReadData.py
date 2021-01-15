import numpy as np 
import pandas as pd 

from troch.utils.data import Dataset,DataLoader

class MyData(Dataset):
    def __init__(self,X,Y):
        self.x=X.values
        self.y=Y

    def __getitem__(self,idx):
        return (self.x[idx],self.y[idx])

    def __len__(self):
        return len(self.y) 