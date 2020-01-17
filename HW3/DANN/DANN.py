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
from torch.autograd import Variable

from tensorboardX import SummaryWriter
# from test import evaluate
from sklearn.metrics import accuracy_score

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


def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)    



if __name__=='__main__':

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

    if args.adaptation== '0' and args.dataset_train== args.dataset_test and args.dataset_train=='m' :
        print('Upper bound of MNIST-M')
    elif args.adaptation== '0' and args.dataset_train== args.dataset_test:
        print('Upper bound of SVHN')
    elif args.adaptation== '0' and args.dataset_train=='m' :
        print('Lower bound of SVHN')
    elif args.adaptation== '0' :
        print('Lower bound of MNIST-M')
    elif args.dataset_train=='m':
        print('Domain adaptation MNIST-M ==> SVHN')
    elif args.dataset_train=='s':
        print('Domain adaptation SVHN ==> MNIST-M')
    else:
        print('dataset error')
        exit()


    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train', dataset= args.dataset_train, adaptation=args.adaptation),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=False)
    val_loader  = torch.utils.data.DataLoader(data.DATA(args, mode='test', dataset=args.dataset_test, adaptation='0'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=False)
    # train_loader_svhn   = torch.utils.data.DataLoader(data.DATA(args, mode='train', dataset='svhn'),
    #                                            batch_size=args.train_batch, 
    #                                            num_workers=args.workers,
    #                                            shuffle=True)
    # val_loader_svhn     = torch.utils.data.DataLoader(data.DATA(args, mode='test', dataset='svhn'),
    #                                            batch_size=args.train_batch, 
    #                                            num_workers=args.workers,
    #                                            shuffle=False)
    # ''' load model '''
    print('===> prepare model ...')
    model = models.dann(args)
    model.cuda() # load model to gpu

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()
    criterion_domain = nn.NLLLoss()

    ''' setup optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    
    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' train model '''
    if args.adaptation=='0':

        print('===> start training ...')
        iters = 0
        best_acc = 0
        for epoch in range(1, args.epoch+1):
            
            model.train()
            
            for idx, (imgs, cls) in enumerate(train_loader):
                
                train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))
                iters += 1

                ''' move data to gpu '''
                imgs, cls = imgs.cuda(), cls.cuda()
                
                ''' forward path '''
                output, _ = model(imgs)

                ''' compute loss, backpropagation, update parameters '''
                # print(output)
                # print(cls)
                # exit()
                loss = criterion(output, cls) # compute loss
                
                optimizer.zero_grad()         # set grad of all parameters to zero
                loss.backward()               # compute gradient for each parameters
                optimizer.step()              # update parameters

                ''' write out information to tensorboard '''
                writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
                train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            # print(train_info)
        
            if epoch%args.val_epoch == 0:
                ''' evaluate the model '''
                acc = evaluate(model, val_loader)        
                writer.add_scalar('val_acc', acc, iters)
                print('Epoch: [{}] ACC:{}'.format(epoch, acc))
                
                ''' save best model '''
                if acc > best_acc:
                    save_model(model, os.path.join(args.save_dir, 'model_best_adtaptation_{}_src_{}_tgt_{}..pth.tar'.format( args.adaptation, args.dataset_train, args.dataset_test)))
                    best_acc = acc

            ''' save model '''
            save_model(model, os.path.join(args.save_dir, 'model_epoch_{}_adtaptation_{}_src_{}_tgt_{}.pth.tar'.format(epoch, args.adaptation, args.dataset_train, args.dataset_test)))

    if args.adaptation == '1':

        print('===> start training ...')
        iters = 0
        best_acc = 0
        for epoch in range(1, args.epoch+1):
            
            model.train()
            
            for idx, (imgs_src, imgs_tgt, cls) in enumerate(train_loader):
                
                train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))
                
                p = float(idx + epoch * len(train_loader)) / args.epoch+1 / len(train_loader)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                
                iters += 1

                imgs_src, imgs_tgt, cls = imgs_src.cuda(), imgs_tgt.cuda(), cls.cuda()    

                class_output, domain_output = model(input_data=imgs_src, alpha=alpha) 

                err_s_label = criterion(class_output, cls)
                err_s_domain = criterion_domain(domain_output, Variable(torch.zeros(args.train_batch).long().cuda()))

                _ , domain_output = model(input_data=imgs_tgt, alpha=alpha)

                err_t_domain = criterion_domain(domain_output, Variable(torch.ones(args.train_batch).long().cuda()))

                loss = err_t_domain + err_s_domain + err_s_label

                optimizer.zero_grad()         # set grad of all parameters to zero
                loss.backward()               # compute gradient for each parameters
                optimizer.step()              # update parameters


            if epoch%args.val_epoch == 0:
                ''' evaluate the model '''
                acc = evaluate(model, val_loader)        
                writer.add_scalar('val_acc', acc, iters)
                print('Epoch: [{}] ACC:{}'.format(epoch, acc))
                
                ''' save best model '''
                if acc > best_acc:
                    save_model(model, os.path.join(args.save_dir, 'model_best_adtaptation_{}_src_{}_tgt_{}.pth.tar'.format(args.adaptation, args.dataset_train, args.dataset_test)))
                    best_acc = acc

            ''' save model '''
            save_model(model, os.path.join(args.save_dir, 'model_epoch_{}_adtaptation_{}_src_{}_tgt_{}.pth.tar'.format(epoch, args.adaptation, args.dataset_train, args.dataset_test)))



                