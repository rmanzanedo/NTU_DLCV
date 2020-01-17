import torch
import torch.nn as nn
import torchvision
import numpy as np



class Resnet(nn.Module):

    def __init__(self, args):
        super(Resnet, self).__init__()
        self.resn = torchvision.models.resnet50(pretrained='imagenet')
        # self.model = nn.Sequential(*list(resn.children())[:-1])

        self.model = nn.Linear(1000, 2048)

    def forward(self, x):
        x1 = self.resn(x)
        # x1 = self.model(x)
        x2 = self.model(x1)
        # x3 = self.model2(x2)

        return x2


class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()

        # self.model1 = nn.Sequential(*list(model_conv.children())[9])


        # Self.model.fc = nn.Linear(1000, 2048)

        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel

        self.model1 = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 11),
            nn.Softmax(1)
        )
    def forward(self, x):
        x1 = self.model1(x)
        # x2 = self.model2(x1)
        # x3 = self.model2(x2)

        return x1