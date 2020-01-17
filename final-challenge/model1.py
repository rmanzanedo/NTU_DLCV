import torch.nn as nn
import torchvision

from torch_summary import summary
import parser

class Resnet50_bn_local(nn.Module):

    def __init__(self):
        super(Resnet50_bn_local, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        # self.resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
        self.resnet50_local = nn.Sequential(*list(resnet50.children())[:-1])

        self.flat = nn.Flatten()





    def forward(self, x_local):


        y_local = self.flat(self.resnet50_local(x_local))

        return y_local


class Resnet50_bn(nn.Module):

    def __init__(self):
        super(Resnet50_bn, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
        # self.resnet50_local = nn.Sequential(*list(resnet50.children())[:-1])

        self.flat = nn.Flatten()





    def forward(self, x_global):

            x = self.resnet50(x_global)
            y = self.flat(x)
            return y


class classiffier(nn.Module):

    def __init__(self):
        super(classiffier, self).__init__()

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(4096),
            nn.Linear(2048 * 1 * 1, 72),
            nn.Softmax(1)
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features=512  * 4, out_features=72),
        #     nn.Softmax(dim=1)
        # )
    def forward(self, feat):

        z = self.classifier(feat)

        return z


class Densenet121_local(nn.Module):

    def __init__(self):
        super(Densenet121_local, self).__init__()
        # densenet121 = torchvision.models.densenet121(pretrained=True)
        # self.densenet121 = nn.Sequential(*list(densenet121.children())[:-1])

        densenet121_local = torchvision.models.densenet121(pretrained=True)
        self.densenet121_local = nn.Sequential(*list(densenet121_local.children())[:-1])


        self.extractor_local = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Flatten()
        )

        # self.norm = nn.BatchNorm1d(512 * 2 * 4)



    def forward(self, x_local):
        # if (mode == 'train'):
        #     x_global = self.densenet121(x_global)
        #     y_global = self.extractor_global(x_global)
            x_local = self.densenet121_local(x_local)
            y_local = self.extractor_local(x_local)
            # y = self.norm(y_global + y_local)
            # z = self.classifier(y)
            return y_local


class Densenet121_global(nn.Module):

    def __init__(self):
        super(Densenet121_global, self).__init__()
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.densenet121 = nn.Sequential(*list(densenet121.children())[:-1])



        self.extractor_global = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Flatten()
        )




    def forward(self, x_global):
        # if (mode == 'train'):
        x_global = self.densenet121(x_global)
        y_global = self.extractor_global(x_global)
        # x_local = self.densenet121_local(x_local)
        # y_local = self.extractor_local(x_local)
        # y = self.norm(y_global + y_local)
        # z = self.classifier(y)
        return y_global


class vgg16_local(nn.Module):
    def __init__(self):
        super(vgg16_local, self).__init__()
        # self.vgg = torchvision.models.vgg16(pretrained=True).features
        self.vgg_local = torchvision.models.vgg16(pretrained=True).features

        self.extractor = nn.Sequential(

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        )

        self.flat = nn.Flatten()
        # self.norm = nn.BatchNorm1d(2048 * 2 * 4)

        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features=2048 * 2 * 4, out_features=72),
        #     nn.Softmax(dim=1)
        # )

    def forward(self, x_local):
        # if (mode == 'train'):
            # x_global = self.vgg(x_global)
            # y_global = self.flat(self.extractor(x_global))
        y_local = self.flat(self.vgg_local(x_local))

        # y = self.norm(y_global + y_local)
        # z = self.classifier(y)
        return y_local
        # else:
        #     x_global = self.vgg(x_global)
        #     x = self.extractor(x_global)
        #     y = self.flat(x)
        #     return y


class vgg16_global(nn.Module):
    def __init__(self):
        super(vgg16_global, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True).features
        # self.vgg_local = torchvision.models.vgg16(pretrained=True).features

        self.extractor = nn.Sequential(

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        )

        self.flat = nn.Flatten()
        # self.norm = nn.BatchNorm1d(2048 * 2 * 4)
        #
        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features=2048 * 2 * 4, out_features=72),
        #     nn.Softmax(dim=1)
        # )

    def forward(self, x_global):
        # if (mode == 'train'):
        x_global = self.vgg(x_global)
        y_global = self.flat(self.extractor(x_global))
            # y_local = self.flat(self.vgg_local(x_local))
            #
            # y = self.norm(y_global + y_local)
            # z = self.classifier(y)
        return y_global
        # else:
        #     x_global = self.vgg(x_global)
        #     x = self.extractor(x_global)
        #     y = self.flat(x)
        #     return y

class classiffier_vgg(nn.Module):

    def __init__(self):
        super(classiffier_vgg, self).__init__()

        # self.classifier = nn.Sequential(
        #     nn.BatchNorm1d(2048),
        #     nn.Linear(2048 * 1 * 1, 72),
        #     nn.Softmax(1)
        # )

        self.classifier = nn.Sequential(
            nn.Linear(in_features= 8192, out_features=72),
            nn.Softmax(dim=1)
        )
    def forward(self, feat):

        z = self.classifier(feat)

        return z