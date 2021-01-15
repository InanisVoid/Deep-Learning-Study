from torch.utils.tensorboard import SummaryWriter 
writer = SummaryWriter("logs")
for i in range(100):
    writer.add_scalar("y_2x",2*i,i)
writer.close()