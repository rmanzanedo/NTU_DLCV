import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data as t_data
import torchvision.datasets as datasets
from torchvision import transforms
from tensorboardX import SummaryWriter
import os

import data
import parser
import models




# d_steps = 100
# g_steps = 100
if __name__=='__main__':

    args = parser.arg_parse()

    # data_transforms = transforms.Compose([transforms.ToTensor()])
    # mnist_trainset = datasets.MNIST(root='./data', train=True,    
    #                            download=True, transform=data_transforms)


    # # batch_size=500
    # train_loader = t_data.DataLoader(mnist_trainset, 
    #                                            batch_size=args.train_batch,
    #                                            shuffle=True
    #                                            )

    print('===> prepare dataloader ...')
    loader = torch.utils.data.DataLoader(data.DATA(args),
                                                   batch_size=args.train_batch, 
                                                   num_workers=args.workers,
                                                   shuffle=True)

    mean = 0.
    meansq = 0.
    for data1 in loader:
        data=data1[0].numpy()
        mean = data.mean()
        meansq = (data**2).mean()

        data2=data1[1].numpy()
        mean2 = data2.mean()
        meansq2 = (data2**2).mean()

        data3=data1[2].numpy()
        mean3 = data3.mean()
        meansq3 = (data3**2).mean()

    std1 = np.sqrt(meansq - mean**2)
    std2 = np.sqrt(meansq2 - mean2**2)
    std3 = np.sqrt(meansq3 - mean3**2)
    print("mean: " + str(mean))
    print("std: " + str(std1))
    print("mean: " + str(mean2))
    print("std: " + str(std2))
    print("mean: " + str(mean3))
    print("std: " + str(std3))
