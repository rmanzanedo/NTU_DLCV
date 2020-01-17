import os
import torch

import parser
import models3 as models
import data_test as data

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv

from sklearn.metrics import accuracy_score

def evaluate(model, data_loader):

    ''' set model to evaluate mode '''
    model.eval()
    # preds = []
    # gts = []
    data_list=[['image_name','label']]
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        for idx, (imgs, name_img) in enumerate(data_loader):
            imgs = imgs.cuda()
            result = model(imgs)
            pred = result[1]#.data.max(1, keepdim=True)[1]
            
            _, pred = torch.max(pred, dim = 1)

            pred = pred.cpu().numpy().squeeze()
            # gt = gt.numpy().squeeze()
            # print(pred,gt)
            # exit()
            # print(gt,pred)
            # exit()
            for i in range(len(name_img)):
                data_list.append([name_img[i], pred[i]])
    #         preds.append(pred)
    #         gts.append(gt)

    # gts = np.concatenate(gts)
    # preds = np.concatenate(preds)

    # return accuracy_score(gts, preds)

    with open(args.save_csv, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(data_list)

    # return accuracy_score(gts, preds)

if __name__ == '__main__':
    
    args = parser.arg_parse()

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(data.DATA(args, mode='test'),
                                              batch_size=args.test_batch, 
                                              num_workers=args.workers,
                                              shuffle=False)
    ''' prepare mode '''
    model = models.SVHNmodel().cuda()

    ''' resume save model '''
    # if args.model == 'mnistm':
    #     checkpoint = torch.load('DANN/log/model_best_adtaptation_1_src_s_tgt_m.pth.tar')
    #     model.load_state_dict(checkpoint)
    # elif args.model == 'svhn':
    #     checkpoint = torch.load('DANN/log/model_best_adtaptation_1_src_m_tgt_s.pth.tar')
    #     model.load_state_dict(checkpoint)
    # else:
    #     print('error to load de model')
    # print(args.load_model)
    checkpoint = torch.load(args.load_model)
    model.load_state_dict(checkpoint)
    
    acc = evaluate(model, test_loader)

    print('CSV created')