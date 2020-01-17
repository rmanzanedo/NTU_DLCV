import os
import torch

import parser
import models
import data_graph as data

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score




if __name__ == '__main__':
    
    args = parser.arg_parse()

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    m_loader = torch.utils.data.DataLoader(data.DATA(args, mode='test'),
                                              batch_size=10000, 
                                              num_workers=args.workers,
                                              shuffle=False)
    s_loader = torch.utils.data.DataLoader(data.DATA(args, mode='test',dataset='s'),
                                              batch_size=26032, 
                                              num_workers=args.workers,
                                              shuffle=False)
    print('===> prepare model ...')
    ''' prepare mode '''
    model = models.dann(args).cuda()

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
    print('===> T-SNE...')
    # acc = evaluate(model, test_loader)
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        for idx, (imgs, cls) in enumerate(m_loader):

            imgs = imgs.cuda()

            result = model(imgs,graph=1)

            # print(result.cpu().numpy().shape)
            # exit()
    X_m = TSNE(n_components=2).fit_transform(result.cpu().numpy())

    cls_m=cls.numpy()
    data_list=X_m.tolist()
    for i in range(len(cls_m)):
        data_list[i].append(cls_m[i])

    with open('mnistm.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(data_list)



    # print('next')
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        for idx1, (imgs1, cls1) in enumerate(s_loader):

            imgs1 = imgs1.cuda()

            result1 = model(imgs1,graph=1)
    
    X_s = TSNE(n_components=2).fit_transform(result1.cpu().numpy())

    cls_s = cls1.numpy()    

    data_list1=X_s.tolist()
    
    for i in range(len(cls_s)):
        data_list1[i].append(cls_s[i])

    with open('svhn.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(data_list1)

    # print('===> Graph ...')
    # fig, ax = plt.subplots()
    # scatter = ax.scatter(X_m[:,0],X_m[:,1],c=cls_m , cmap='tab10')
    # scatter = ax.scatter(X_s[:,0],X_s[:,1],c=cls_s , cmap='tab10')

    # legend1 = ax.legend(*scatter.legend_elements(),
    #                 loc="Upper right", title="Classes")
    # plt.savefig('1.png')
    # plt.close()


    # fig, ax = plt.subplots()
    # scatter = ax.scatter(X_m[:,0],X_m[:,1],c=1 )
    # scatter = ax.scatter(X_s[:,0],X_s[:,1],c=2 )

    # legend1 = ax.legend(*scatter.legend_elements(),
    #                 loc="Upper right", title="Domain")
    # plt.savefig('2.png')
    # plt.close()
    # for i in range(len(cls_m)):
    #     plt.scatter(X_m[i][0],X_m[i][1],c=cls_m[i] , alpha=0.7)
    # for j in range(len(cls_s)):
    #     plt.scatter(X_s[j][0],X_s[j][1],c=cls_s[j] , alpha=0.7)
    print('CSV created')