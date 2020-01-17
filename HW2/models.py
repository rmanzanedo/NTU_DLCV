import torch
import torch.nn as nn
import torchvision
import numpy as np

class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()
        model_conv = torchvision.models.resnet18(pretrained='imagenet')
        self.model=nn.Sequential(*list(model_conv.children())[:-2])
        
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.tconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1,bias=False)
        self.relu1 = nn.ReLU()
        
        self.tconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1,bias=False)
        self.relu2 = nn.ReLU()
        
        self.tconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1,bias=False)
        self.relu3 = nn.ReLU()
        
        self.tconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1,bias=False)
        self.relu4 = nn.ReLU()
        
        self.tconv5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1,bias=False)
        self.relu5 = nn.ReLU()
        
        self.conv1 = nn.Conv2d(16, 9, kernel_size=1, stride=1, padding=0)
        
        
        
        
    def forward(self, x):
            
        x1 = self.model(x)    
        x2 = self.relu1(self.tconv1(x1))
        x3 = self.relu2(self.tconv2(x2))
        x4 = self.relu3(self.tconv3(x3))
        x5 = self.relu4(self.tconv4(x4))
        x6 = self.relu5(self.tconv5(x5))
        x7 = (self.conv1(x6))
        
#        y0 = x.cpu().detach().numpy()
#        np.save('model0.npy',y0)
#        y1 = x1.cpu().detach().numpy()
#        np.save('model1.npy',y1)
#        y2 = x2.cpu().detach().numpy()
#        np.save('model2.npy',y2)
#        y3 = x3.cpu().detach().numpy()
#        np.save('model3.npy',y3)
#        y4 = x4.cpu().detach().numpy()
#        np.save('model4.npy',y4)
#        y5 = x5.cpu().detach().numpy()
#        np.save('model5.npy',y5)
#        y6 = x6.cpu().detach().numpy()
#        np.save('model6.npy',y6)
#        y7 = x7.cpu().detach().numpy()
#        np.save('model7.npy',y7)
        
        #print(type(y))
#        return torch.softmax(x7,dim=1)
        return x7
