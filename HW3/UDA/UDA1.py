import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
from models import DSN
# from data_loader import GetLoader 
from functions import SIMSE, DiffLoss, MSE
# from test import test
import data
import parser
from sklearn.metrics import accuracy_score
import csv
# torch.multiprocessing.set_sharing_strategy('file_system')


def evaluate(model, data_loader):

    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    gts = []
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        for idx, (imgs, gt) in enumerate(data_loader):
            imgs = imgs.cuda()
            result = my_net(input_data=imgs, mode='source', rec_scheme='share')
            pred = result[3]#.data.max(1, keepdim=True)[1]
            
            _, pred = torch.max(pred, dim = 1)

            pred = pred.cpu().numpy().squeeze()
            gt = gt.numpy().squeeze()
            
            preds.append(pred)
            gts.append(gt)

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)
    # print(gts)
    # print(preds)
    # exit()
    data_list=[]
    for i in range(len(gts)):
        data_list.append([gts[i],preds[i]])
    with open('output.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(data_list)

    return accuracy_score(gts, preds)

args = parser.arg_parse()

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

######################
# params             #
######################

# source_image_root = os.path.join('.', 'dataset', 'mnist')
# target_image_root = os.path.join('.', 'dataset', 'mnist_m')
# model_root = 'model'
cuda = True
cudnn.benchmark = True
lr = args.lr     #1e-2
batch_size = args.train_batch #32
image_size = 28
n_epoch = args.epoch
step_decay_weight = 0.95
lr_decay_step = 20000
active_domain_loss_step = 10000
weight_decay = 1e-6
alpha_weight = 0.01
beta_weight = 0.075
gamma_weight = 0.25
momentum = 0.9

# manual_seed = random.randint(1, 10000)
# random.seed(manual_seed)
# torch.manual_seed(manual_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)


#######################
# load data           #
#######################
print('===> prepare dataloader ...')

# img_transform = transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ])

# dataset_source = datasets.MNIST(
#     root=source_image_root,
#     train=True,
#     transform=img_transform
# )

# dataloader_source = torch.utils.data.DataLoader(
#     dataset=dataset_source,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=8
# )

# train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')

# dataset_target = GetLoader(
#     data_root=os.path.join(target_image_root, 'mnist_m_train'),
#     data_list=train_list,
#     transform=img_transform
# )

# dataloader_target = torch.utils.data.DataLoader(
#     dataset=dataset_target,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=8
# )
train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train', dataset= args.dataset_train, adaptation=args.adaptation),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=False)
val_loader  = torch.utils.data.DataLoader(data.DATA(args, mode='test', dataset=args.dataset_test, adaptation='0'),
                                           batch_size=args.train_batch, 
                                           num_workers=args.workers,
                                           shuffle=False)
#####################
#  load model       #
#####################
print('===> prepare model ...')
my_net = DSN()

#####################
# setup optimizer   #
#####################


def exp_lr_scheduler(optimizer, step, init_lr=lr, lr_decay_step=lr_decay_step, step_decay_weight=step_decay_weight):

    # Decay learning rate by a factor of step_decay_weight every lr_decay_step
    current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

    if step % lr_decay_step == 0:
        print('learning rate is set to %f' % current_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return optimizer


optimizer = optim.Adam(my_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)  #optim.SGD(my_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

loss_classification = nn.CrossEntropyLoss()
loss_recon1 = MSE()
loss_recon2 = SIMSE()
loss_diff = DiffLoss()
loss_similarity = nn.CrossEntropyLoss()

# if cuda:
my_net = my_net.cuda()
loss_classification = loss_classification.cuda()
loss_recon1 = loss_recon1.cuda()
loss_recon2 = loss_recon2.cuda()
loss_diff = loss_diff.cuda()
loss_similarity = loss_similarity.cuda()

for p in my_net.parameters():
    p.requires_grad = True

#############################
# training network          #
#############################


len_dataloader = len(train_loader)
# dann_epoch = np.floor(active_domain_loss_step / len(train_loader))

current_step = 0
for epoch in range(1, args.epoch+1):

    # data_source_iter = iter(dataloader_source)
    # data_target_iter = iter(dataloader_target)

    # i = 0

    # while i < len_dataloader:
    for idx, (imgs_src, imgs_tgt, cls) in enumerate(train_loader):

        ###################################
        # target data training            #
        ###################################

        train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))

        # data_target = data_target_iter.next()
        t_img, t_label = imgs_tgt, cls

        my_net.zero_grad()
        loss = 0
        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        # input_img.resize_as_(t_img).copy_(t_img)
        # class_label.resize_as_(t_label).copy_(t_label)
        target_inputv_img = Variable(input_img)
        target_classv_label = Variable(class_label)
        target_domainv_label = Variable(domain_label)

        if current_step > active_domain_loss_step:
            # p = float(idx + (epoch - dann_epoch) * len_dataloader / (n_epoch - dann_epoch) / len_dataloader)
            # p = 2. / (1. + np.exp(-10 * p)) - 1
            p = float(idx + epoch * len(train_loader)) / args.epoch+1 / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # activate domain loss
            result = my_net(input_data=target_inputv_img, mode='target', rec_scheme='all', p=p)
            target_privte_code, target_share_code, target_domain_label, target_rec_code = result
            target_dann = gamma_weight * loss_similarity(target_domain_label, target_domainv_label)
            loss += target_dann
        else:
            target_dann = Variable(torch.zeros(1).float().cuda())
            result = my_net(input_data=target_inputv_img, mode='target', rec_scheme='all')
            target_privte_code, target_share_code, _, target_rec_code = result

        target_diff= beta_weight * loss_diff(target_privte_code, target_share_code)
        loss += target_diff
        target_mse = alpha_weight * loss_recon1(target_rec_code, target_inputv_img)
        loss += target_mse
        target_simse = alpha_weight * loss_recon2(target_rec_code, target_inputv_img)
        loss += target_simse

        # loss.backward()
        # optimizer.step()

        ###################################
        # source data training            #
        ###################################

        # data_source = data_source_iter.next()
        s_img, s_label = imgs_src, cls

        my_net.zero_grad()
        batch_size = len(s_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()

        loss = 0

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(input_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)
        source_inputv_img = Variable(input_img)
        source_classv_label = Variable(class_label)
        source_domainv_label = Variable(domain_label)

        if current_step > active_domain_loss_step:

            # activate domain loss

            result = my_net(input_data=source_inputv_img, mode='source', rec_scheme='all', p=p)
            source_privte_code, source_share_code, source_domain_label, source_class_label, source_rec_code = result
            source_dann = gamma_weight * loss_similarity(source_domain_label, source_domainv_label)
            loss += source_dann
        else:
            source_dann = Variable(torch.zeros(1).float().cuda())
            result = my_net(input_data=source_inputv_img, mode='source', rec_scheme='all')
            source_privte_code, source_share_code, _, source_class_label, source_rec_code = result

        source_classification = loss_classification(source_class_label, source_classv_label)
        loss += source_classification

        source_diff = beta_weight * loss_diff(source_privte_code, source_share_code)
        loss += source_diff
        source_mse = alpha_weight * loss_recon1(source_rec_code, source_inputv_img)
        loss += source_mse
        source_simse = alpha_weight * loss_recon2(source_rec_code, source_inputv_img)
        loss += source_simse

        loss.backward()
        optimizer = exp_lr_scheduler(optimizer=optimizer, step=current_step)
        optimizer.step()

        # i += 1
        current_step += 1
    # print 'source_classification: %f, source_dann: %f, source_diff: %f, ' \
    #       'source_mse: %f, source_simse: %f, target_dann: %f, target_diff: %f, ' \
    #       'target_mse: %f, target_simse: %f' \
    #       % (source_classification.data.cpu().numpy(), source_dann.data.cpu().numpy(), source_diff.data.cpu().numpy(),
    #          source_mse.data.cpu().numpy(), source_simse.data.cpu().numpy(), target_dann.data.cpu().numpy(),
    #          target_diff.data.cpu().numpy(),target_mse.data.cpu().numpy(), target_simse.data.cpu().numpy())

    # print 'step: %d, loss: %f' % (current_step, loss.cpu().data.numpy())
    # torch.save(my_net.state_dict(), model_root + '/dsn_mnist_mnistm_epoch_' + str(epoch) + '.pth')
    # test(epoch=epoch, name='mnist')
    # test(epoch=epoch, name='mnist_m')
        print(train_info, end='\r')
    if epoch%args.val_epoch == 0:
                ''' evaluate the model '''
                acc = evaluate(my_net, val_loader)        
                # writer.add_scalar('val_acc', acc, iters)
                print(train_info)
                print('Epoch: [{}] ACC:{}'.format(epoch, acc))

print('done')