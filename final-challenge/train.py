import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
import parser
import numpy as np
import sys
import pickle
import torch_summary
from test import *
from loss_utils import *
from model import *
from data import *

if __name__=='__main__':

    sys.stdout.flush()
    torch.cuda.empty_cache()
    args = parser.arg_parse()

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' load dataset and prepare data loader '''
    print('===> LOADING DATA ...')
    args = parser.arg_parse()
    train_data = Data(args, mode="train")
    
    ''' load model '''
    print('===> LOADING MODEL ...')
    if (args.selected_model == 'Resnet18_bn'):
        model = Resnet18()
    elif (args.selected_model == 'Resnet34_bn'):
        model = Resnet34_bn()
    elif (args.selected_model == 'Resnet50_bn'):
        model = Resnet50_bn()
    elif (args.selected_model == 'vgg16'):
        model = vgg16()
    elif (args.selected_model == 'Densenet121'):
        model = Densenet121()
    elif (args.selected_model == 'Densenet121_local'):
        model = Densenet121_local()
    elif (args.selected_model == 'vgg16_local'):
        model = vgg16_local()
    elif (args.selected_model == 'Resnet18_bn_local'):
        model = Resnet18_bn_local()
    elif (args.selected_model == 'Resnet34_bn_local'):
        model = Resnet34_bn_local()
    elif (args.selected_model == 'Resnet50_bn_local'):
        model = Resnet50_bn_local()
    else:
        print('Change model!')

    model.cuda()
    
    if (args.features == 'local'):
        #summary(model.cuda(), [(3,256,512),(3,128, 256), 'train'])
        print(model)
    else:
        summary (model, (3, 256, 256))
    
    ''' define loss '''
    CE_Loss = CrossEntropyLabelSmooth(72)
    triplet_Loss = nn.TripletMarginLoss(margin=1.0, p=2)
    
    ''' setup optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    ''' train model '''
    print(args.selected_model, ' ===> START TRAINING FOR '+ str(args.epoch) + ' EPOCHS')
    loss_list_epochs = []
    rank1_acc_list = []
    best_rank1_acc = 0.0
    model.train()
    running_loss = 0
    count = 0
    epoch = 1
    running_CE_loss = 0
    running_triplet_loss = 0
    CE_loss_list_epochs = []
    triplet_loss_list_epochs = []
    
    print('===> START TRAINING FOR EPOCH: ' + str(epoch))
    print('Learning rate: ', args.lr)
    
    while(train_data.completed_epochs < args.epoch):
        if (train_data.completed_epochs == epoch-1):
            
            imgs, local_imgs, labels, _ = train_data.next()
            
            if(labels == []):
                continue
            
            labels = torch.LongTensor(np.array(labels))
            count += 1
            
            ''' move data to gpu '''
            imgs, labels = imgs.cuda(), labels.cuda()
            
            if (args.features == 'local'):
                local_imgs = local_imgs.cuda()
            
            optimizer.zero_grad()
            
            ''' forward, compute loss, backpropagation, update parameters '''
            if (args.features == 'local'):
                features, output = model(imgs, local_imgs, 'train')
            else:
                features, output = model(imgs)
            
            #Triplet
            dist_mat = euclidean_dist(features, features)
            dist_ap, dist_an, p_ind, n_ind = compute_distances_indices(dist_mat, labels)
            
            tri_loss = triplet_Loss(features, features[p_ind,:], features[n_ind,:])
            
            #Label Smoothing CrossEntropy
            loss_CE = CE_Loss(output, labels) 
            running_CE_loss += loss_CE.item()
            running_triplet_loss  += tri_loss.item()
            
            #Sum the 2 Loss
            loss = loss_CE + tri_loss
            
            running_loss += loss.item()
            loss.backward()
            optimizer.step()            

        else:
            
            # validation stage
            model.eval()
            print('===> START VALIDATION FOR EPOCH '+ str(epoch))
            rank1_acc, names = evaluate(model, args)
            
            #print loss and training info, save results for the current epoch

            print('===> EPOCH: ', str(epoch), ' Loss =', str(running_loss/count), ' CELoss =', str(running_CE_loss/count), 'TripletLoss =', str(running_triplet_loss/count), ' Rank-1 validation accuracy=', str(rank1_acc))
            CE_loss_list_epochs.append(running_CE_loss/count)
            triplet_loss_list_epochs.append(running_triplet_loss/count)
            running_CE_loss = 0
            running_triplet_loss = 0

            loss_list_epochs.append(running_loss/count)    
            rank1_acc_list.append(rank1_acc)
            
            if(rank1_acc > best_rank1_acc):
                best_rank1_acc = rank1_acc
                print('New best rank-1 accuracy = ', best_rank1_acc)
                print('===> SAVING BEST MODEL')
                model_name = 'best_model_' + args.selected_model + '.pth.tar'
                torch.save(model.state_dict(), os.path.join(args.save_dir, model_name))
                
            #initialize next epoch
            epoch += 1
            print('===> START TRAINING FOR EPOCH: ' + str(epoch))
            #adaptive learning rate
            # if(epoch <= 10):
            #     for g in optimizer.param_groups:
            #         g['lr'] = 3.5e-5 * epoch/10
            #     print('Learning rate: ', g['lr'])
            # elif(epoch <= 70 and epoch > 10):
            #     for g in optimizer.param_groups:
            #         g['lr'] = 3.5e-5
            #     print('Learning rate: ', g['lr'])
            # else:
            #     for g in optimizer.param_groups:
            #         g['lr'] = 3.5e-6
            #     print('Learning rate: ', g['lr'])
            
            running_loss = 0
            count = 0
    filename = 'Loss_epochs_' + args.selected_model
    with open('Loss_epochs', 'wb') as fle:
        pickle.dump(loss_list_epochs, fle)



