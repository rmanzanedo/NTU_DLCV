import torch
import torch.nn as nn
import torchvision
import numpy as np


# class generator(nn.Module):
    
#     def __init__(self, inp, out):
        
#         super(generator, self).__init__()
        
#         self.net = nn.Sequential(
#                                  nn.Linear(inp,300),
#                                  nn.BatchNorm1d(300),
#                                  nn.ReLU(inplace=True),
#                                  nn.Linear(300,1000),
#                                  nn.BatchNorm1d(1000),
#                                  nn.ReLU(inplace=True),
#                                  nn.Linear(1000,800),
#                                  nn.BatchNorm1d(800),
#                                  nn.ReLU(inplace=True),
#                                  nn.Linear(800,out)
#                                     )
        
#     def forward(self, x):
#         x = self.net(x)
#         return x
# class generator(torch.nn.Module):
#     """
#     A three hidden-layer generative neural network
#     """
#     def __init__(self, inp, out):
#         super(generator, self).__init__()
#         n_features = inp
#         n_out = out
        
#         self.hidden0 = nn.Sequential(
#             nn.Linear(n_features, 256),
#             nn.LeakyReLU(0.2)
#         )
#         self.hidden1 = nn.Sequential(            
#             nn.Linear(256, 512),
#             nn.LeakyReLU(0.2)
#         )
#         self.hidden2 = nn.Sequential(
#             nn.Linear(512, 1024),
#             nn.LeakyReLU(0.2)
#         )
        
#         self.out = nn.Sequential(
#             nn.Linear(1024, n_out),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         x = self.hidden0(x)
#         x = self.hidden1(x)
#         x = self.hidden2(x)
#         x = self.out(x)
#         return x
# defining discriminator class

# class discriminator(nn.Module):
    
#     def __init__(self, inp, out):
        
#         super(discriminator, self).__init__()
        
#         self.net = nn.Sequential(
#                                  nn.Linear(inp,300),
#                                  nn.BatchNorm1d(300),
#                                  nn.LeakyReLU(inplace=True),
#                                  nn.Linear(300,300),
#                                  nn.BatchNorm1d(300),
#                                  nn.LeakyReLU(inplace=True),
#                                  nn.Linear(300,200),
#                                  nn.BatchNorm1d(200),
#                                  nn.LeakyReLU(inplace=True),
#                                  nn.Linear(200,out),
#                                  nn.Sigmoid()
#                                     )
        
#     def forward(self, x):
#         x = self.net(x)
#         return x

# class discriminator(torch.nn.Module):
#     """
#     A three hidden-layer discriminative neural network
#     """
#     def __init__(self,  inp, out):
#         super(discriminator, self).__init__()
#         n_features = inp
#         n_out = out
        
#         self.hidden0 = nn.Sequential( 
#             nn.Linear(n_features, 1024),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3)
#         )
#         self.hidden1 = nn.Sequential(
#             nn.Linear(1024, 512),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3)
#         )
#         self.hidden2 = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3)
#         )
#         self.out = nn.Sequential(
#             torch.nn.Linear(256, n_out),
#             torch.nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.hidden0(x)
#         x = self.hidden1(x)
#         x = self.hidden2(x)
#         x = self.out(x)
#         return x



class generator(nn.Module):
    def __init__(self, _):
        super(generator, self).__init__()
        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( 100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # state size. (64*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. (64*4) x 8 x 8
            nn.ConvTranspose2d( 64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. (64*2) x 16 x 16
            nn.ConvTranspose2d( 64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (64) x 32 x 32
            nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (3) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)




class discriminator(nn.Module):
    def __init__(self, _):
        super(discriminator, self).__init__()
        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (3) x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 32 x 32
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*8) x 4 x 4
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



