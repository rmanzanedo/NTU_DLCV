import os
import torch

import parser
import models
import data_test as data
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
    # train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
    #                                            batch_size=args.train_batch,
    #                                            num_workers=args.workers,
    #                                            shuffle=True)
    val_loader = torch.utils.data.DataLoader(data.DATA(args),
                                             batch_size=args.train_batch,
                                             num_workers=args.workers,
                                             shuffle=False)
    ''' load model '''
    print('===> prepare model ...')
    ResNet = models.Resnet(args)
    model = models.Net(args)

    checkpoint = torch.load(args.load_model)
    model.load_state_dict(checkpoint)

    ResNet.cuda()
    ResNet.eval()
    model.eval()

     # load model to gpu

    ''' define loss '''


    ''' setup optimizer '''


    ''' setup tensorboard '''


    ''' train model '''
    print('===> creating the data ...')

    # tensor = torch.ones((2,), dtype=torch.long)

    preds=[]
    gts = []
    # x=0
    f = open(args.save_txt+"p1_valid.txt", "w+")
    with torch.no_grad():
        for idx, (imgs, gt) in enumerate(val_loader):

            train_info = 'Epoch: [{0}/{1}]'.format( idx + 1, len(val_loader))

            ''' move data to gpu '''

            imgs = imgs.squeeze().type(torch.FloatTensor).cuda()
            # print('len:', imgs.shape)


            ''' forward path '''
            output = torch.reshape(torch.mean(ResNet(imgs).cpu(), 0), (1, 2048))
            # print(output.squeeze().shape)
            output1 = model(torch.reshape(output.squeeze(), (1,2048)))

            _, pred = torch.max(output1, dim=1)

            pred = np.array(pred.cpu().numpy())
            gt = gt.numpy()

            preds.append(pred)
            gts.append(gt)
            print(train_info)
            if idx+1 == len(val_loader):
                f.write(str(pred[0]))
            else:
                f.write(str(pred[0]) + '\n')
            # x+=1
            # if x==4:
            #     f.close()
            #     exit()

        gts = np.concatenate(gts)
        preds = np.concatenate(preds)

        f.close()
        result = accuracy_score(gts, preds)
        print('ACC : {}'.format(result))




