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
from test1 import *
from loss_utils import *
from model1 import *
from data import *

if __name__ == '__main__':

    sys.stdout.flush()
    torch.cuda.empty_cache()
    args = parser.arg_parse()

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> LOADING DATA ...')
    args = parser.arg_parse()
    train_data = Data(args, mode="train")

    ''' load model '''
    print('===> LOADING MODEL ...')

    if (args.selected_model == 'rn'):
        model_global = Resnet50_bn()
        model_local = Resnet50_bn_local()
        model_class = classiffier()
    elif (args.selected_model == 'dn'):
        model_global = Densenet121_global()
        model_local = Densenet121_local()
        model_class = classiffier()
    elif (args.selected_model == 'dn-rn'):
        model_global = Densenet121_global()
        model_local = Resnet50_bn_local()
        model_class = classiffier()
    elif (args.selected_model == 'rn-dn'):
        model_global = Resnet50_bn()
        model_local = Densenet121_local()
        model_class = classiffier()
    elif (args.selected_model == 'vgg'):
        model_global = vgg16_global()
        model_local = vgg16_local()
        model_class = classiffier_vgg()
    # elif (args.selected_model == 'Densenet121_local'):
    #     model = Densenet121_local()
    # elif (args.selected_model == 'vgg16_local'):
    #     model = vgg16_local()
    # elif (args.selected_model == 'Resnet18_bn_local'):
    #     model = Resnet18_bn_local()
    # elif (args.selected_model == 'Resnet34_bn_local'):
    #     model = Resnet34_bn_local()
    # elif (args.selected_model == 'Resnet50_bn_local'):
    #     model = Resnet50_bn_local()
    else:
        print('Change model!')  
        

    # model_global = Densenet121_global()
    # model_local = Resnet50_bn_local()
    # model_class = classiffier()

    model_global.cuda()
    model_local.cuda()
    model_class.cuda()

    #if (args.features == 'local'):
        # summary(model.cuda(), [(3,256,512),(3,128, 256), 'train'])
    #    print(model)
    #else:
    #    summary(model, (3, 256, 512))

    ''' define loss '''
    CE_Loss = CrossEntropyLabelSmooth(72)
    triplet_Loss = nn.TripletMarginLoss(margin=1.0, p=2)

    ''' setup optimizer '''
    optimizer_global = torch.optim.Adam(model_global.parameters(), lr=args.lr)
    optimizer_local = torch.optim.Adam(model_local.parameters(), lr=args.lr)
    optimizer_class = torch.optim.Adam(model_class.parameters(), lr=args.lr)

    ''' train model '''
    print(args.selected_model, ' ===> START TRAINING FOR ' + str(args.epoch) + ' EPOCHS')
    loss_list_epochs = []
    rank1_acc_list = []
    best_rank1_acc = 0.0
    model_global.train()
    model_local.train()
    model_class.train()
    running_loss = 0
    count = 0
    epoch = 1
    running_CE_loss = 0
    running_triplet_loss = 0
    running_triplet_loss_local = 0
    CE_loss_list_epochs = []
    triplet_loss_list_epochs = []
    triplet_loss_list_epochs_local=[]


    print('===> START TRAINING FOR EPOCH: ' + str(epoch))
    print('Learning rate: ', args.lr)

    while (train_data.completed_epochs < args.epoch):
        if (train_data.completed_epochs == epoch - 1):

            imgs, local_imgs, labels, _ = train_data.next()

            if (labels == []):
                continue

            labels = torch.LongTensor(np.array(labels))
            count += 1

            ''' move data to gpu '''
            imgs, labels = imgs.cuda(), labels.cuda()

            # if (args.features == 'local'):
            local_imgs = local_imgs.cuda()



            ''' forward, compute loss, backpropagation, update parameters '''
            # if (args.features == 'local'):
            #     features, output = model(imgs, local_imgs, 'train')
            # else:
            #     features, output = model(imgs)
            global_feat = model_global(imgs)

            local_feat = model_local(local_imgs)

            output = model_class(local_feat + global_feat)

            # global Triplet
            dist_mat = euclidean_dist(global_feat, global_feat)
            dist_ap, dist_an, p_ind, n_ind = compute_distances_indices(dist_mat, labels)

            tri_loss = triplet_Loss(global_feat, global_feat[p_ind, :], global_feat[n_ind, :])

            # local Triplet
            dist_mat_local = euclidean_dist(local_feat, local_feat)
            dist_ap, dist_an, p_ind_local, n_ind_local = compute_distances_indices(dist_mat_local, labels)

            tri_loss_local = triplet_Loss(local_feat, local_feat[p_ind_local, :], local_feat[n_ind_local, :])

            # Label Smoothing CrossEntropy
            loss_CE = CE_Loss(output, labels)
            running_CE_loss += loss_CE.item()
            running_triplet_loss += tri_loss.item()
            running_triplet_loss_local +=tri_loss_local.item()

            # Sum the 2 Loss
            global_loss = loss_CE + tri_loss
            local_loss = loss_CE + tri_loss_local

            running_loss += global_loss.item()

            # training global feat
            optimizer_global.zero_grad()
            global_loss.backward(retain_graph=True)
            optimizer_global.step()

            # training local feat
            optimizer_local.zero_grad()
            local_loss.backward(retain_graph=True)
            optimizer_local.step()

            # training the classifier
            optimizer_class.zero_grad()
            loss_CE.backward(retain_graph=True)
            optimizer_class.step()


        else:

            # validation stage
            model_global.eval()
            print('===> START VALIDATION FOR EPOCH ' + str(epoch))
            rank1_acc, names = evaluate(model_global, args)

            # print loss and training info, save results for the current epoch

            print('===> EPOCH: ', str(epoch), ' Loss =', str(running_loss / count), ' CELoss =',
                  str(running_CE_loss / count), 'TripletLoss =', str(running_triplet_loss / count),
                  ' Rank-1 validation accuracy=', str(rank1_acc))
            CE_loss_list_epochs.append(running_CE_loss / count)
            triplet_loss_list_epochs.append(running_triplet_loss / count)
            triplet_loss_list_epochs_local.append(running_triplet_loss_local / count)
            running_CE_loss = 0
            running_triplet_loss = 0
            running_triplet_loss_local = 0

            loss_list_epochs.append(running_loss / count)
            rank1_acc_list.append(rank1_acc)

            if (rank1_acc > best_rank1_acc):
                best_rank1_acc = rank1_acc
                print('New best rank-1 accuracy = ', best_rank1_acc)
                print('===> SAVING BEST MODEL')
                model_name = 'best_model_' + args.selected_model + '.pth.tar'
                torch.save(model_global.state_dict(), os.path.join(args.save_dir, model_name))

            # initialize next epoch
            epoch += 1
            print('===> START TRAINING FOR EPOCH: ' + str(epoch))
            # adaptive learning rate
            list_opt=[optimizer_global,optimizer_local,optimizer_class]
            for optimizer in list_opt:
                if (epoch <= 10):
                    for g in optimizer.param_groups:
                        g['lr'] = 3.5e-5 * epoch / 10
                    print('Learning rate: ', g['lr'])
                elif (epoch <= 20 and epoch > 10):
                    for g in optimizer.param_groups:
                        g['lr'] = 3.5e-5
                    print('Learning rate: ', g['lr'])
                elif (epoch <= 30 and epoch > 20):
                    for g in optimizer.param_groups:
                        g['lr'] = 3.5e-5 * (epoch-20) / 10
                    print('Learning rate: ', g['lr'])
                elif (epoch <= 40 and epoch > 30):
                    for g in optimizer.param_groups:
                        g['lr'] = 3.5e-7
                    print('Learning rate: ', g['lr'])
                elif (epoch <= 110 and epoch > 100):
                    for g in optimizer.param_groups:
                        g['lr'] = 3.5e-7
                    print('Learning rate: ', g['lr'])
                else:
                    for g in optimizer.param_groups:
                        g['lr'] = 3.5e-6
                    print('Learning rate: ', g['lr'])

                # if (epoch <= 10):
                #     for g in optimizer.param_groups:
                #         g['lr'] = 3.5e-5 * epoch / 10
                #     print('Learning rate: ', g['lr'])
                # elif (epoch <= 20 and epoch > 10):
                #     for g in optimizer.param_groups:
                #         g['lr'] = 3.5e-5
                #     print('Learning rate: ', g['lr'])
                # elif (epoch <= 30 and epoch > 20):
                #     for g in optimizer.param_groups:
                #         g['lr'] = 3.5e-6
                #     print('Learning rate: ', g['lr'])
                # elif (epoch <= 40 and epoch > 30):
                #     for g in optimizer.param_groups:
                #         g['lr'] = 3.5e-7
                #     print('Learning rate: ', g['lr'])
                # else:
                #     for g in optimizer.param_groups:
                #         g['lr'] = 3.5e-6
                #     print('Learning rate: ', g['lr'])

            running_loss = 0
            count = 0
    filename = 'log/loss/Loss_epochs_' + args.selected_model
    with open(filename, 'wb') as fle:
        # pickle.dump(loss_list_epochs, fle)
        pickle.dump(triplet_loss_list_epochs, fle)
        pickle.dump(triplet_loss_list_epochs_local, fle)
        pickle.dump(CE_loss_list_epochs, fle)
