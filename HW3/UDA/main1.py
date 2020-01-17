from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
# import sys
# import itertools
# import logging
# from dataset_mnist import *
# from dataset_usps import *
# from net_config import *
# from optparse import OptionParser
import data
import parser
import numpy as np
from sklearn.metrics import accuracy_score
import os


# Training settings
# parser = OptionParser()
# parser.add_option('--config',
#                   type=str,
#                   help="net configuration",
#                   default="usps2mnist.yaml")

# (opts, args) = parser.parse_args(sys.argv)
# config = NetConfig(opts.config)
# kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
# torch.manual_seed(config.seed)
# if torch.cuda.is_available() == False:
#     use_cuda = False
#     print("invalid cuda access") 
# if use_cuda:
#     torch.cuda.manual_seed(config.seed)

args = parser.arg_parse()
use_cuda = True
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)


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
        x_f = self.fc1(x.view(-1, 320))
        x = F.dropout(F.relu(x_f), training=self.training)
        x = self.fc2(x)
        return x_f, F.log_softmax(x, dim=1)

class Discrimer(nn.Module):
    def __init__(self):
        super(Discrimer, self).__init__()
        self.fc1 = nn.Linear(50, 512)
        self.fc2 = nn.Linear(512, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()
model_src = Net()
critic = Discrimer()

if use_cuda:
    model.cuda()
    model_src.cuda()
    critic.cuda()

optimizer_d = optim.Adam(critic.parameters(), lr=args.lr)
optimizer_g = optim.Adam(model.parameters(), lr=args.lr)
print("load model...")
# PATH = config.pretrained_path #'pytorch_model_usps2mnist'    
model.load_state_dict(torch.load('log/model_best_adtaptation_1_src_m_tgt_s.pth.tar')) #model for adapt
model_src.load_state_dict(torch.load('log/model_best_adtaptation_1_src_m_tgt_s.pth.tar'))


def train(epoch):
    model.train()
    
    # for batch_idx, ((data_src, target_src), (data, target)) in enumerate(itertools.izip(train_loader_a, train_loader_b)):
    for idx, (imgs_src, imgs_tgt, cls) in enumerate(train_loader):

        ###################################
        # target data training            #
        ###################################

        train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))
        if use_cuda:
            data_src, target_src = imgs_src.cuda(), cls.cuda()
            data = imgs_tgt.cuda()
        data_src, target_src = Variable(data_src), Variable(target_src)
        data = Variable(data)
       
        feat_src, output_src = model_src(data_src)
        feat, output = model(data)
        all_d_feat = torch.cat((feat_src,feat),0)
        all_d_score = critic(all_d_feat)
        # print(type(all_d_score.size()[0]))
        # exit()
        all_d_label = torch.cat((Variable(torch.ones(100).long().cuda()),Variable(torch.zeros(100).long().cuda())),0)
        #D loss
        domain_loss = F.nll_loss(all_d_score, all_d_label)
        ###domain accuracy###
        predict = torch.squeeze(all_d_score.max(1)[1])
        d_accu = (predict == all_d_label).float().mean()

        critic.zero_grad()
        model.zero_grad()
        domain_loss.backward#(retain_variables=True)
        optimizer_d.step()
        
        #G loss
        gen_loss = F.nll_loss(all_d_score[100:], Variable(torch.ones(100).long().cuda()))
        
        model.zero_grad()
        critic.zero_grad()
        gen_loss.backward()
        optimizer_g.step()
        print(train_info, end='\r')

    if epoch%args.val_epoch == 0:
        ''' evaluate the model '''
        acc = evaluate(model, test_loader_b)        
        # writer.add_scalar('val_acc', acc, iters)
        print(train_info)
        print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            # if acc > best_acc:
            #         save_model(model, os.path.join(args.save_dir, 'model_best_adtaptation_{}_src_{}_tgt_{}.pth.tar'.format(args.adaptation, args.dataset_train, args.dataset_test)))
            #         best_acc = acc
            #         exit()
        
        # if batch_idx % config.log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG Loss: {:.6f}\tD Loss: {:.6f}\tD accu: {:.3f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader_a.dataset),
            #     100. * batch_idx / len(train_loader_a), gen_loss.data[0],domain_loss.data[0],d_accu.data[0]))
        

def evaluate(model, data_loader):

    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    gts = []
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        for idx, (imgs, gt) in enumerate(data_loader):
            imgs = imgs.cuda()
            pred, _ = model(imgs)
            
            _, pred = torch.max(pred, dim = 1)

            pred = pred.cpu().numpy().squeeze()
            gt = gt.numpy().squeeze()
            
            preds.append(pred)
            gts.append(gt)

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    return accuracy_score(gts, preds)
# def test(epoch):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader_b:
#         if use_cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data, volatile=True), Variable(target)
#         feat, output = model(data)
#         target = torch.squeeze(target)
#         test_loss += F.nll_loss(output, target).data[0]
#         pred = output.data.max(1)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data).cpu().sum()

#     test_loss = test_loss
#     test_loss /= len(test_loader_b) # loss function already averages over batch size
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader_b.dataset),
#         100. * correct / len(test_loader_b.dataset)))


for epoch in range(1, args.epoch + 1):
    train(epoch)    
    # test(epoch)