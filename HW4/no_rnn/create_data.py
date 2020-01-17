import os
import torch

import parser
import models
import data
import test

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
# from test import evaluate





if __name__ == '__main__':

    args = parser.arg_parse()

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(data.DATA(args, mode='valid'),
                                             batch_size=args.train_batch,
                                             num_workers=args.workers,
                                             shuffle=True)
    ''' load model '''
    print('===> prepare model ...')
    ResNet = models.Resnet(args)

    ResNet.cuda()
    ResNet.eval()

     # load model to gpu

    ''' define loss '''


    ''' setup optimizer '''


    ''' setup tensorboard '''


    ''' train model '''
    print('===> creating the data ...')

    # tensor = torch.ones((2,), dtype=torch.long)

    preds=[]
    gts = []
    with torch.no_grad():
        for idx, (imgs, cls) in enumerate(train_loader):

            train_info = 'Epoch: [{0}/{1}]'.format( idx + 1, len(train_loader))

            ''' move data to gpu '''

            imgs = imgs.squeeze().type(torch.FloatTensor).cuda()
            # print('len:', imgs.shape)


            ''' forward path '''
            output = torch.reshape(torch.mean(ResNet(imgs).cpu(), 0), (1, 2048))

            preds.append(output)
            gts.append(cls.item())

            ''' compute loss, backpropagation, update parameters '''


            ''' write out information to tensorboard '''


            print(train_info)


    pred = torch.stack(preds)
    real = torch.LongTensor(gts)
    torch.save(pred, 'train_resnet.pt')
    torch.save(real, 'train_cls.pt')




