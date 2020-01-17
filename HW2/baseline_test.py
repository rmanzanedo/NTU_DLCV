# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 19:25:04 2019

@author: User
"""


import os
import torch

import parser1
import models
import data1


import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image  
import PIL 


from sklearn.metrics import accuracy_score


def evaluate_test(model, data_loader):

    ''' set model to evaluate mode '''
    model.eval()
#    preds = []
#    gts = []
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        for idx, (imgs,fname) in enumerate(data_loader):
            imgs = imgs.cuda()
            pred = model(imgs)
#            print(fname)
            
            _, pred = torch.max(pred, dim = 1)

            pred = pred.cpu().numpy().squeeze()
            # print(pred.shape)
#            gt = gt.numpy().squeeze()
            
#            preds.append(pred)
#            gts.append(gt)
            
            for img in range(len(fname)):
                
                Image.fromarray(np.uint8(pred[img])).save(os.path.join(args.save_dir,fname[img])) 

#    gts = np.concatenate(gts)
#    preds = np.concatenate(preds)

#    return mean_iou_score(preds, gts)

if __name__ == '__main__':
    
    args = parser1.arg_parse()

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(data1.DATA(args),
                                              batch_size=args.test_batch, 
                                              num_workers=args.workers,
                                              shuffle=True)
    ''' prepare mode '''
    model = models.Net(args).cuda()
#    model_std = torch.load(os.path.join(args.save_dir, 'model_best.pth.tar'))#'model_1.pth.tar')
#    model.load_state_dict(model_std)
#    model = torch.nn.DataParallel(model,
#                                  device_ids=list(range(torch.cuda.device_count()))).cuda()

    ''' resume save model '''
    checkpoint = torch.load(args.resume)
#    checkpoint = torch.load(os.path.join(args.save_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint)

    acc = evaluate_test(model, test_loader)
    print('done')