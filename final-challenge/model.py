import torch.nn as nn
import torchvision

from torch_summary import summary
import parser


class Resnet18_bn(nn.Module):

    def __init__(self):
        super(Resnet18_bn, self).__init__()
        resnet18 = torchvision.models.resnet18(pretrained=False)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
        
        self.flat = nn.Flatten()
        
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(512*1*1),
            nn.ReLU(),
            nn.Linear(512*1*1, 72),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.resnet18(x)
        y = self.flat(x)
        z = self.model(x)

        return y, z

class Resnet18_bn_local(nn.Module):

    def __init__(self):
        super(Resnet18_bn_local, self).__init__()
        resnet18 = torchvision.models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
        self.resnet18_local = nn.Sequential(*list(resnet18.children())[:-1])
        
        self.flat = nn.Flatten()
        
        self.norm = nn.BatchNorm1d(512)
        
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512*1*1, 72),
            nn.Softmax(1)
        )
    
    def forward(self, x_global, x_local, mode):
        if (mode == 'train'):
            y_global = self.flat(self.resnet18(x_global))
            y_local = self.flat(self.resnet18_local(x_local))
            y = self.norm(y_global + y_local)
            z = self.classifier(y)
            return y,z 
        else:
            x = self.resnet18(x_global)
            y = self.flat(x)
            return y 
    
    
class Resnet50_bn(nn.Module):

    def __init__(self):
        super(Resnet50_bn, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=False)
        self.resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
        
        self.flat = nn.Flatten()
        
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(2048*1*1),
            nn.ReLU(),
            nn.Linear(2048*1*1, 72),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.resnet50(x)
        y = self.flat(x)
        z = self.model(x)

        return y, z

class Resnet50_bn_local(nn.Module):

    def __init__(self):
        super(Resnet50_bn_local, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
        self.resnet50_local = nn.Sequential(*list(resnet50.children())[:-1])
        
        self.flat = nn.Flatten()
        
        self.norm = nn.BatchNorm1d(2048)
        
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2048*1*1, 72),
            nn.Softmax(1)
        )
    
    def forward(self, x_global, x_local, mode):
        if (mode == 'train'):
            y_global = self.flat(self.resnet50(x_global))
            y_local = self.flat(self.resnet50_local(x_local))
            y = self.norm(y_global + y_local)
            z = self.classifier(y)
            return y,z 
        else:
            x = self.resnet50(x_global)
            y = self.flat(x)
            return y    

    
class Resnet34_bn(nn.Module):

    def __init__(self):
        super(Resnet34_bn, self).__init__()
        resnet34 = torchvision.models.resnet34(pretrained=False)
        self.resnet34 = nn.Sequential(*list(resnet34.children())[:-1])
        
        self.flat = nn.Flatten()
        
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(512*1*1),
            nn.ReLU(),
            nn.Linear(512*1*1, 72),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.resnet34(x)
        y = self.flat(x)
        z = self.model(x)

        return y, z

class Resnet34_bn_local(nn.Module):

    def __init__(self):
        super(Resnet34_bn_local, self).__init__()
        resnet34 = torchvision.models.resnet34(pretrained=True)
        self.resnet34 = nn.Sequential(*list(resnet34.children())[:-1])
        self.resnet34_local = nn.Sequential(*list(resnet34.children())[:-1])
        
        self.flat = nn.Flatten()
        
        self.norm = nn.BatchNorm1d(512)
        
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512*1*1, 72),
            nn.Softmax(1)
        )
    
    def forward(self, x_global, x_local, mode):
        if (mode == 'train'):
            y_global = self.flat(self.resnet34(x_global))
            y_local = self.flat(self.resnet34_local(x_local))
            y = self.norm(y_global + y_local)
            z = self.classifier(y)
            return y,z 
        else:
            x = self.resnet34(x_global)
            y = self.flat(x)
            return y
    
class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True).features
        
        self.flat = nn.Flatten()
        
        self.model = nn.Sequential(

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Flatten(),
            nn.Linear(in_features=2048*2*4, out_features=72),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.vgg(x)
        y = self.flat(x)
        z = self.model(x)

        return y,z

class vgg16_local(nn.Module):
    def __init__(self):
        super(vgg16_local, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True).features
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
        self.norm = nn.BatchNorm1d(2048*2*4)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048*2*4, out_features=72),
            nn.Softmax(dim=1)
            )
        
    def forward(self, x_global, x_local, mode):
        if (mode == 'train'):
            x_global = self.vgg(x_global)
            y_global = self.flat(self.extractor(x_global))
            y_local = self.flat(self.vgg_local(x_local))
            
            y = self.norm(y_global + y_local)
            z = self.classifier(y)
            return y,z 
        else:
            x_global = self.vgg(x_global)
            x = self.extractor(x_global)
            y = self.flat(x)
            return y
   
    
class Densenet121(nn.Module):

    def __init__(self):
        super(Densenet121, self).__init__()
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.densenet121 = nn.Sequential(*list(densenet121.children())[:-1])
        
        self.flat = nn.Flatten()
        
        self.model = nn.Sequential(

            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
            nn.Flatten(),
            nn.Linear(in_features=2048*2*4, out_features=72),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.densenet121(x)
        y = self.flat(x)
        z = self.model(x)
        return y,z     
    
class Densenet121_local(nn.Module):

    def __init__(self):
        super(Densenet121_local, self).__init__()
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.densenet121 = nn.Sequential(*list(densenet121.children())[:-1])
        
        densenet121_local = torchvision.models.densenet121(pretrained=True)
        self.densenet121_local = nn.Sequential(*list(densenet121_local.children())[:-1])
        
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
        
        self.norm = nn.BatchNorm1d(512*2*2)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*2*2, out_features=72),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x_global, x_local, mode):
        if (mode == 'train'):
            x_global = self.densenet121(x_global)
            y_global = self.extractor_global(x_global)
            x_local = self.densenet121_local(x_local)
            y_local = self.extractor_local(x_local)
            y = self.norm(y_global + y_local)
            z = self.classifier(y)
            return y,z 
        else:
            x_global = self.densenet121(x_global)
            y = self.extractor_global(x_global)
            return y
    
if __name__=="__main__":
    args = parser.arg_parse()
    model = Resnet18_bn()
    print('Resnet18 with batch normalization model:')
    summary(model.cuda(), (3,256,512))
    model = Resnet50_bn()
    print('\n\nResnet50 with batch normalization model:')
    summary(model.cuda(), (3,256,512))
    model = Resnet34_bn()
    print('\n\nResnet34 with batch normalization model:')
    summary(model.cuda(), (3,256,512))
    model = vgg16()
    print('\n\nvgg16 model:')
    summary(model.cuda(), (3,256,512))
    model = Densenet121()
    print('\n\nDensenet121 model:')
    summary(model.cuda(), (3,256,512))
