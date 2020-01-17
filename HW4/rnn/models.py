import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable



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

class rnn(nn.Module):

    def __init__(self, input_size=2048, hidden_size=512):
        super(rnn, self).__init__()

        # self.rnn = nn.GRU(input_size, num_layers=2, hidden_size=hidden_size, batch_first=True,dropout=1)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.5)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 11),
            nn.Softmax(1)
        )

    def forward(self, x,len=0):
    # def forward(self, embedded) :
    #     hidden = self.init_hidden()

        # Pack them up nicely
        # embedded = nn.utils.rnn.pack_padded_sequence(x,len)
        # propagate input through RNN

        # out, hidden = self.rnn(x, None)
        out, (hn,cn) = self.rnn(x, None)

        # unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out)
        #
        # indices = Variable(torch.LongTensor(np.array(unpacked_len) - 1).view(-1, 1)
        #                    .expand(unpacked.size(0), unpacked.size(2))
        #                    .unsqueeze(1))
        # print(indices.shape)
        # print(unpacked.shape)
        # print(out[0].shape)
        # print(hidden)
        out = self.fc(hn[-1])

        # out = self.fc(out[0])
        # if use_gpu:
        #     out = out.cuda()

        return out, hn[-1]

    # def init_hidden(self, seq_len=32):
    #     # if use_gpu:
    #     #     return Variable(torch.zeros(num_layers, seq_len, hidden_size), requires_grad=True).cuda()
    #
    #     return Variable(torch.zeros(2, seq_len, 128), requires_grad=True).cuda()