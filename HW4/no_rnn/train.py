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


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def evaluate(model, data_loader, labels, args):
    ''' set model to evaluate mode '''
    model.eval()

    preds = []
    # gts = []
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        for idx in range(0,len(data_loader),args.train_batch):

            imgs = data_loader[idx:idx + args.train_batch].squeeze().cuda()
            # gt = np.array(labels[idx:idx + args.train_batch])
            # output = torch.reshape(torch.mean(ResNet(imgs), 0), (1, 2048))
            if imgs.shape[0]== 2048:
                pred = model(torch.reshape(imgs, (1,2048)))
            else:
                pred = model(imgs)
            _, pred = torch.max(pred, dim=1)

            pred = np.array(pred.cpu().numpy())
            # gt = gt.numpy()

            preds.append(pred)
            # gts.append(gt)

    # gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    return accuracy_score(labels, preds)


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
    # val_loader = torch.utils.data.DataLoader(data.DATA(args, mode='valid'),
    #                                          batch_size=args.train_batch,
    #                                          num_workers=args.workers,
    #                                          shuffle=True)
    train_loader = torch.load('train_resnet.pt')
    train_cls = torch.load('train_cls.pt')

    val_loader = torch.load ('valid_resnet.pt')
    val_cls = torch.load('valid_cls.pt')
    ''' load model '''
    print('===> prepare model ...')
    # ResNet = models.Resnet(args)
    model = models.Net(args)
    # ResNet.cuda()
    # ResNet.eval()

    model.cuda()  # load model to gpu

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()

    ''' setup optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' train model '''
    print('===> start training ...')
    iters = 0
    best_acc = 0
    # tensor = torch.ones((2,), dtype=torch.long)

    for epoch in range(1, args.epoch + 1):
        model.train()

        for idx in range(0,len(train_loader),args.train_batch):

            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, (int(idx/args.train_batch) + 1), round(len(train_loader)/args.train_batch))
            iters += 1
            ''' move data to gpu '''
            # print(imgs.shape)
            # if idx+args.train_batch > len(train_loader):
            #     # break
            #     imgs = train_loader[idx:].squeeze().cuda()
            #     cls = train_cls[idx:].cuda()
            # else:
            imgs = train_loader[idx:idx+args.train_batch].squeeze().cuda()
            cls =train_cls[idx:idx+args.train_batch].cuda()

            # imgs = imgs

            ''' forward path '''
            # output = torch.reshape(torch.mean(ResNet(imgs).cpu(), 0), (1, 2048))
            # print(output.cpu().shape)
            # exit()

            output = model(imgs)

            # tensor = torch.ones((imgs.shape[0],), dtype=torch.long)
            # cls1 = tensor.fill_( cls.item()).cuda()
            # cls1 = torch.full((imgs.shape[0],), cls.item(), dtype=torch.long).cuda()
            ''' compute loss, backpropagation, update parameters '''
            loss = criterion(torch.reshape(output,(imgs.shape[0],11)), cls.long()) # compute loss

            optimizer.zero_grad()  # set grad of all parameters to zero
            loss.backward()  # compute gradient for each parameters
            optimizer.step()  # update parameters

            ''' write out information to tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print('\r',train_info, end='' )

        if epoch % args.val_epoch == 0:
            ''' evaluate the model '''
            acc = evaluate(model, val_loader, val_cls,args)
            writer.add_scalar('val_acc', acc, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            ''' save best model '''
            if acc > best_acc:
                save_model(model, os.path.join(args.save_dir, 'model_best.pth.tar'))
                best_acc = acc

        ''' save model '''
        save_model(model, os.path.join(args.save_dir, 'model_{}.pth.tar'.format(epoch)))