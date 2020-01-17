import torch
import torch.nn as nn
import torchvision
import numpy as np


# class generator(nn.Module):
#     def __init__(self):
#         super(generator, self).__init__()

#         self.label_emb = nn.Embedding(1, 100)

#         self.init_size = 64 // 4  # Initial size before upsampling
#         self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(128),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, 3, stride=1, padding=1),
#             nn.BatchNorm2d(128, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 64, 3, stride=1, padding=1),
#             nn.BatchNorm2d(64, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 3, 3, stride=1, padding=1),
#             nn.Tanh(),
#         )

#     def forward(self, noise, labels):
#         gen_input = torch.mul(self.label_emb(labels), noise)
#         out = self.l1(gen_input)
#         print(out.size())
#         print(out.shape[0])
#         out = out.view(10, 128, self.init_size, self.init_size)
#         print(out.size())
#         exit()
#         img = self.conv_blocks(out)
#         return img


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        # self.ngpu = ngpu
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( 101, 64 * 8, 4, 1, 0, bias=False),
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
            # nn.Dropout(0.2),
            # state size. (64) x 32 x 32
            nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (3) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# # defining discriminator class

# class discriminator(nn.Module):
#     def __init__(self):
#         super(discriminator, self).__init__()

#         def discriminator_block(in_filters, out_filters, bn=True):
#             """Returns layers of each discriminator block"""
#             block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
#             if bn:
#                 block.append(nn.BatchNorm2d(out_filters, 0.8))
#             return block

#         self.conv_blocks = nn.Sequential(
#             *discriminator_block(3, 16, bn=False),
#             *discriminator_block(16, 32),
#             *discriminator_block(32, 64),
#             *discriminator_block(64, 128),
#         )

#         # The height and width of downsampled image
#         ds_size = 64 // 2 ** 4

#         # Output layers
#         self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
#         self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Softmax())

#     def forward(self, img):
#         out = self.conv_blocks(img)
#         out = out.view(out.shape[0], -1)
#         validity = self.adv_layer(out)
#         label = self.aux_layer(out)

#         return validity, label


import torch.nn as nn


# custom weights initialization called on netG and discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# class generator(nn.Module):

#     def __init__(self):

#         super(generator, self).__init__()
#         self.ReLU = nn.ReLU(True)
#         self.Tanh = nn.Tanh()
#         self.conv1 = nn.ConvTranspose2d(101, 64 * 8, 4, 1, 0, bias=False)
#         self.BatchNorm1 = nn.BatchNorm2d(64 * 8)

#         self.conv2 = nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False)
#         self.BatchNorm2 = nn.BatchNorm2d(64 * 4)

#         self.conv3 = nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False)
#         self.BatchNorm3 = nn.BatchNorm2d(64 * 2)

#         self.conv4 = nn.ConvTranspose2d(64 * 2, 64 * 1, 4, 2, 1, bias=False)
#         self.BatchNorm4 = nn.BatchNorm2d(64 * 1)

#         self.conv5 = nn.ConvTranspose2d(64 * 1, 3, 4, 2, 1, bias=False)

#         self.apply(weights_init)


#     def forward(self, input):

#         x = self.conv1(input)
#         x = self.BatchNorm1(x)
#         x = self.ReLU(x)

#         x = self.conv2(x)
#         x = self.BatchNorm2(x)
#         x = self.ReLU(x)

#         x = self.conv3(x)
#         x = self.BatchNorm3(x)
#         x = self.ReLU(x)

#         x = self.conv4(x)
#         x = self.BatchNorm4(x)
#         x = self.ReLU(x)

#         x = self.conv5(x)
#         output = self.Tanh(x)
#         return output

    
class discriminator(nn.Module):

    def __init__(self):

        super(discriminator, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(64 * 2)
        self.conv3 = nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(64 * 4)
        self.conv4 = nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False)
        self.BatchNorm4 = nn.BatchNorm2d(64 * 8)
        self.conv5 = nn.Conv2d(64 * 8, 64 * 1, 4, 1, 0, bias=False)
        self.BatchNorm5 = nn.BatchNorm2d(64)
        self.disc_linear = nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False)
        self.aux_linear = nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

        # self.64 = 64
        self.apply(weights_init)

    def forward(self, input):

        x = self.conv1(input)
        x = self.LeakyReLU(x)

        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.LeakyReLU(x)

        x = self.conv3(x)
        x = self.BatchNorm3(x)
        x = self.LeakyReLU(x)

        x = self.conv4(x)
        x = self.BatchNorm4(x)
        x = self.LeakyReLU(x)

        # x = self.conv5(x)
        # x = self.BatchNorm5(x)
        # x = self.LeakyReLU(x)

        # x = x.view(-1, 64 * 1)
        c = self.aux_linear(x)
        c = self.sigmoid1(c)
        s = self.disc_linear(x)
        s = self.sigmoid2(s)
        return s,c
