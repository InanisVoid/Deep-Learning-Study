import torch 
import torchvision
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn 


train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
train_loader=torch.utils.data.DataLoader(
    train_set,batch_size=10
)


# #Show one batch
# batch=next(iter(train_loader))
# images,labels=batch
# grid = torchvision.utils.make_grid(images,nrow=10)
# plt.figure(figsize=(15,15))
# plt.imshow(np.transpose(grid,(1,2,0)))
# print(labels)
# plt.show()

