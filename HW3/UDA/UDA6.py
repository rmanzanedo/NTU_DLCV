
import os
import os.path as osp
import sys
import argparse
import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score


import models3 as models
import data, parser


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
            # print(pred,gt)
            # exit()
            
            preds.append(pred)
            gts.append(gt)

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    return accuracy_score(gts, preds)

def fit(model, optim, data_loader, val_loader, n_epochs=10, walker_weight = 1., visit_weight = .1 ):

    # train, val = dataset
    # cuda=1
    # cudafy = lambda x : x if cuda is None else x.cuda(cuda)
    # torch2np = lambda x : x.cpu().detach().numpy()

    DA_loss  = models.AssociativeLoss(walker_weight=walker_weight, visit_weight=visit_weight)
    CL_loss  = nn.CrossEntropyLoss()

    model = model.cuda()
    model.train()

    print('training start!')
    # start_time = time.time()

    num_iter = 0
    train_hist = []
    best_acc=0
    # pbar_epoch = tqdm.tqdm()
    # tic = time.time()
    for epoch in range(n_epochs):
        # epoch_start_time = time.time()

        # pbar_batch = tqdm.tqdm(dual_iterator(*dataset))

        for idx, (imgs_src, imgs_tgt, cls) in enumerate(data_loader):

        ###################################
        # target data training            #
        ###################################

            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(data_loader))

            xs = imgs_src.cuda()
            ys = cls.view(-1).cuda()
            xt = imgs_tgt.cuda()

            losses = {}

            ### D CL training
            model.zero_grad()

            phi_s, yp = model(xs)
            phi_t, _ = model(xt)

            class_loss = CL_loss(yp, ys).mean()
            domain_loss = DA_loss(phi_s, phi_t, ys).mean()

            # losses['D acc src']   = torch.eq(yp.max(dim=1)[1], ys).sum().float()  / args.train_batch
            # losses['D acc tgt']   = torch.eq(ypt.max(dim=1)[1], yt).sum().float() / val.batch_size

            (class_loss + domain_loss).backward()
            optim.step()

            # losses = { k : v.cpu().data.detach().numpy() for k, v in losses.items()}
            # losses['batch'] = num_iter
            # train_hist.append(losses)

            num_iter += 1
            print(train_info, end='\r')


        if epoch%args.val_epoch == 0:
            ''' evaluate the model '''
            acc = evaluate(model, val_loader)        
            # writer.add_scalar('val_acc', acc, iters)
            print(train_info)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            if acc > best_acc:
                    save_model(model, os.path.join(args.save_dir, 'model_best_adtaptation_{}_src_{}_tgt_{}.pth.tar'.format( args.adaptation, args.dataset_train, args.dataset_test)))
                    best_acc = acc

            ''' save model '''
            save_model(model, os.path.join(args.save_dir, 'model_epoch_{}_adtaptation_{}_src_{}_tgt_{}.pth.tar'.format(epoch, args.adaptation, args.dataset_train, args.dataset_test)))








if __name__ == '__main__':

    # parser = build_parser()
    args = parser.arg_parse()

    # Network
    
    model   = models.SVHNmodel()
    # model  = models.FrenchModel()

    # Adam optimizer, with amsgrad enabled
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)

    # Dataset
    # datasets = data.load_dataset(path="data", train=True)

    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train', dataset= args.dataset_train, adaptation=args.adaptation),
                                           batch_size=args.train_batch, 
                                           num_workers=args.workers,
                                           shuffle=False)
    val_loader   = torch.utils.data.DataLoader(data.DATA(args, mode='test', dataset=args.dataset_test, adaptation='0'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=False)

    # os.makedirs(args.log, exist_ok=True)
    fit(model, optim, train_loader, val_loader, n_epochs=args.epoch,
               visit_weight=0.1, walker_weight=1)