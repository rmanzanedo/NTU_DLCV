from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import data
import parser
import numpy as np
from sklearn.metrics import accuracy_score
import os
# import sys
# import itertools
# import logging
# from dataset_mnist import *
# from dataset_usps import *
# from net_config import *
# from optparse import OptionParser
# import pdb

# Training settings
# parser = OptionParser()
# parser.add_option('--config',
#                   type=str,
#                   help="net configuration",
#                   default="usps2mnist.yaml")
# (opts, args) = parser.parse_args(sys.argv)
# config = NetConfig(opts.config)
# kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
args = parser.arg_parse()
# torch.manual_seed(args.random_seed)
# if torch.cuda.is_available() == False:
use_cuda = True
#     print("invalid cuda access") 
# if use_cuda:
#     torch.cuda.manual_seed(config.seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)  

def evaluate(model, data_loader):

    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    gts = []
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        for idx, (imgs, gt) in enumerate(data_loader):
            imgs = imgs.cuda()
            result = model(imgs)
            pred = result[1]#.data.max(1, keepdim=True)[1]
            
            _, pred = torch.max(pred, dim = 1)

            pred = pred.cpu().numpy().squeeze()
            gt = gt.numpy().squeeze()
            
            preds.append(pred)
            gts.append(gt)

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)
    # print(gts)
    # print(preds)
    # exit()
    # data_list=[]
    # for i in range(len(gts)):
    #     data_list.append([gts[i],preds[i]])
    # with open('output.csv', 'w', newline='') as file:
    #     writer = csv.writer(file, delimiter=',')
    #     writer.writerows(data_list)

    return accuracy_score(gts, preds)

def read(args):
    # print(config)
    # if os.path.exists(config.log):
    #     os.remove(config.log)
    # base_folder_name = os.path.dirname(config.log)
    # if not os.path.isdir(base_folder_name):
    #     os.mkdir(base_folder_name)
    # logging.basicConfig(filename=config.log, level=logging.INFO, mode='w')
    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # logging.getLogger('').addHandler(console)
    # logging.info("Let the journey begin!")
    # logging.info(config)
    # exec("train_dataset_a = %s(root=config.train_data_a_path, \
    #                               num_training_samples=config.train_data_a_size, \
    #                               train=config.train_data_a_use_train_data, \
    #                               transform=transforms.ToTensor(), \
    #                               seed=config.train_data_a_seed)" % config.train_data_a)
    # train_loader_a = torch.utils.data.DataLoader(dataset=train_dataset_a, batch_size=config.batch_size, shuffle=True)

    # exec("train_dataset_b = %s(root=config.train_data_b_path, \
    #                              num_training_samples=config.train_data_b_size, \
    #                              train=config.train_data_b_use_train_data, \
    #                              transform=transforms.ToTensor(), \
    #                             seed=config.train_data_b_seed)" % config.train_data_b)
    # train_loader_b = torch.utils.data.DataLoader(dataset=train_dataset_b, batch_size=config.batch_size, shuffle=True)

    # exec("test_dataset_b = %s(root=config.test_data_b_path, \
    #                             num_training_samples=config.test_data_b_size, \
    #                             train=config.test_data_b_use_train_data, \
    #                             transform=transforms.ToTensor(), \
    #                             seed=config.test_data_b_seed)" % config.test_data_b)
    # test_loader_b = torch.utils.data.DataLoader(dataset=test_dataset_b, batch_size=config.test_batch_size, shuffle=True)


    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train', dataset= args.dataset_train, adaptation=args.adaptation),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=False)
    val_loader  = torch.utils.data.DataLoader(data.DATA(args, mode='test', dataset=args.dataset_test, adaptation='0'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=False)


    return train_loader, val_loader
    # pdb.set_trace()
    
train_loader,  test_loader_b = read(args)
# best_acc=40

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x_f = F.relu(self.fc1(x))
        x = F.dropout(x_f, training=self.training)
        x = self.fc2(x)
        return x_f, F.log_softmax(x, dim=1)

model = Net()
if use_cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)
def train(epoch):
    model.train()
    best_acc=.40
    # for idx, (data, target) in enumerate(train_loader_a):
    for idx, (imgs_src, imgs_tgt, cls) in enumerate(train_loader):

        ###################################
        # target data training            #
        ###################################

        train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))
        # if use_cuda:
        data, target = imgs_src.cuda(), cls.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        feat, output = model(data)
        target = torch.squeeze(target)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print(train_info, end='\r')
    if epoch%args.val_epoch == 0:
            ''' evaluate the model '''
            acc = evaluate(model, test_loader_b)        
            # writer.add_scalar('val_acc', acc, iters)
            print(train_info)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            if acc > best_acc:
                    save_model(model, os.path.join(args.save_dir, 'model_best_adtaptation_{}_src_{}_tgt_{}.pth.tar'.format(args.adaptation, args.dataset_train, args.dataset_test)))
                    best_acc = acc
                    exit()
        # if idx % config.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, idx * len(data), len(train_loader_a.dataset),
        #         100. * idx / len(train_loader_a), loss.data[0]))

# def test(epoch):
    # model.eval()
    # test_loss = 0
    # correct = 0
    # for data, target in test_loader_b:
    #     if use_cuda:
    #         data, target = data.cuda(), target.cuda()
    #     data, target = Variable(data, volatile=True), Variable(target)
    #     feat, output = model(data)
    #     target = torch.squeeze(target)
    #     test_loss += F.nll_loss(output, target)#.data[0]
    #     pred = output#.data.max(1)[1] # get the index of the max log-probability
    #     correct += pred.eq(target.data).cpu().sum()

    # test_loss = test_loss
    # test_loss /= len(test_loader_b) # loss function already averages over batch size
    # print('\nTest on target valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader_b.dataset),
    #     100. * correct / len(test_loader_b.dataset)))


for epoch in range(1, args.epoch + 1):
    train(epoch)    
    # test(epoch)
    
# PATH = 'pytorch_model_usps2mnist'    
# torch.save(model.state_dict(), PATH)