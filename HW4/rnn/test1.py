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



def input_rnn(images,cls):
    imgs=images
    # lengths = torch.LongTensor([len(imgs[0])])
    #

    # x = nn.utils.rnn.pad_sequence(imgs).cuda()
    # input = nn.utils.rnn.pack_padded_sequence(torch.FloatTensor(imgs[0]).view(lengths.item(),1,2048), lengths).cuda()
    #
    # return torch.reshape(x, (lengths,1, 2048)), label


    lengths = torch.LongTensor(list(map(len, imgs)))

    lengths, idx = lengths.sort(0, descending=True)
    imgs_output = [imgs[i] for i in idx]

    if args.test_batch==1:
        label = torch.LongTensor(cls)
    else:
        label = torch.LongTensor(np.array(cls)[idx])

    x = nn.utils.rnn.pad_sequence(imgs_output).cuda()
    input = nn.utils.rnn.pack_padded_sequence(x, lengths)

    return input, label

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
                                             batch_size=args.test_batch,
                                             num_workers=args.workers,
                                             shuffle=False)
    ''' load model '''
    print('===> prepare model ...')
    ResNet = models.Resnet(args)
    model = models.rnn()

    checkpoint = torch.load(args.load_model)
    model.load_state_dict(checkpoint)

    model.cuda()
    model.eval()

    ResNet.cuda()
    ResNet.eval()

    preds = []
    gts = []
    group=[]
    cls_group=[]
    f = open(args.save_txt + "p2_result.txt", "w+")
    with torch.no_grad():
        for idx, (imgs, cls) in enumerate(val_loader):
            train_info = 'Epoch: [{0}/{1}]'.format(idx + 1, len(val_loader))

            ''' move data to gpu '''
            # iter+=1
            imgs = imgs.squeeze().type(torch.FloatTensor).cuda()
            # print('len:', imgs.shape)

            ''' forward path '''
            output = (ResNet(imgs).cpu())
            # print(output)
            # preds.append(output)
            # gts.append(cls.item())
            #     imgs = train_loader[idx:].squeeze().cuda()
            #     cls = train_cls[idx:].cuda()
            # else:
            # imgs = train_loader[idx:idx+args.train_batch]
            # print(imgs[0].shape[0])
            group.append(output)
            cls_group.append(cls)
            # print(imgs)
            # cls =train_cls[idx:idx+args.train_batch]
            # print(len(group)%5,)
            if len(group)%1==0 or idx+1==len(val_loader):

                x, labels = input_rnn(group, cls_group)
                pred, _ = model(x)


                _, pred = torch.max(pred, dim=1)
                # print(cls,labels,output.cpu())
                pred = np.array(pred.cpu().numpy())
                gt = labels.numpy()



                preds.append(pred)
                gts.append(gt)
                print(train_info)

                for i in range(len(pred)):
                    if idx+1 == len(val_loader) and i==len(pred)-1:
                        f.write(str(pred[i]))
                    else:
                        f.write(str(pred[i]) + '\n')

                group = []
                cls_group = []

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    f.close()
    result = accuracy_score(gts, preds)
    print('ACC : {}'.format(result))




