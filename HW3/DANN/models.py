import torch
import torch.nn as nn
from torch.autograd import Function
# from functions import ReverseLayerF


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class dann(nn.Module):

    def __init__(self, args):
        super(dann, self).__init__()  

    #     ''' declare layers used in this network'''
    #     # first block
    #     self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) # 64x64 -> 64x64
    #     self.bn1 = nn.BatchNorm2d(32)
    #     self.relu1 = nn.ReLU()
    #     self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 64x64 -> 32x32
        
    #     # second block
    #     self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 32x32 -> 32x32
    #     self.bn2 = nn.BatchNorm2d(64)
    #     self.relu2 = nn.ReLU()
    #     self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16
        
    #     # third block
    #     self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # 16x16 -> 16x16
    #     self.bn3 = nn.BatchNorm2d(128)
    #     self.relu3 = nn.ReLU()
    #     self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 -> 8x8

    #     # classification
    #     # self.avgpool = nn.AvgPool2d(16)
    #     # self.fc = nn.Linear(64, 4)
    #     self.avgpool = nn.AvgPool2d(8)
    #     self.fc = nn.Linear(128, 10)

    # def forward(self, img):

    #     x = self.relu1(self.bn1(self.conv1(img)))
    #     x = self.maxpool1(x)
        
    #     x = self.relu2(self.bn2(self.conv2(x)))
    #     x = self.maxpool2(x)
        
    #     x = self.relu3(self.bn3(self.conv3(x)))
    #     x = self.maxpool2(x)

    #     x = self.avgpool(x).view(x.size(0),-1)
    #     x = self.fc(x)

        # return x

        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha=0, graph=0):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        # return torch.softmax(class_output,dim=1)
        if graph:
            return feature
        else:
            return class_output, domain_output