import os
import torch
import data
import parser
import models
import numpy as np
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def input_rnn(images,cls):
    imgs=images
    # lengths = []
    # for i in range(len(imgs)):
    #     lengths.append(imgs[i].shape[0])
    # print(imgs.shape[0])
    lengths = torch.LongTensor(list(map(len, imgs)))

    lengths, idx = lengths.sort(0, descending=True)
    imgs_output = [imgs[i] for i in idx]

    if args.test_batch == 1:
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
    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                               batch_size=args.test_batch,
                                               num_workers=args.workers,
                                               shuffle=True)
    # val_loader = torch.utils.data.DataLoader(data.DATA(args, mode='valid'),
    #                                          batch_size=args.train_batch,
    #                                          num_workers=args.workers,
    #                                          shuffle=True)
    # train_loader = torch.load('train_resnet.pt')
    # train_cls = torch.load('train_cls.pt')
    # print(train_loader.shape)

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
    group = []
    cls_group = []
    with torch.no_grad():
        for idx, (imgs, cls) in enumerate(train_loader):

            train_info = 'Epoch: [{0}/{1}]'.format( idx + 1, len(train_loader))

            ''' move data to gpu '''

            imgs = imgs.squeeze().type(torch.FloatTensor).cuda()
            # print('len:', imgs.shape)


            ''' forward path '''
            output = (ResNet(imgs).cpu())

            # preds.append(output)
            # gts.append(cls.item())
        #     imgs = train_loader[idx:].squeeze().cuda()
        #     cls = train_cls[idx:].cuda()
        # else:
        # imgs = train_loader[idx:idx+args.train_batch]
        # print(imgs[0].shape[0])

        # print(imgs)
        # cls =train_cls[idx:idx+args.train_batch]
            group.append(output)
            cls_group.append(cls)
            x, labels = input_rnn(group, cls_group)
            _, output = model(x)

            pred = np.array(output.cpu().numpy())
            gt = labels.numpy()

            preds.append(pred)
            gts.append(gt)
            group = []
            cls_group = []

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)









    print('===> T-SNE...')
    X_m = TSNE(n_components=2).fit_transform(preds.squeeze())
    x =np.concatenate(X_m[:,0:1])
    y = np.concatenate(X_m[:,1:2])
    fig, ax = plt.subplots(figsize=(30, 17))
    scatter = ax.scatter(x, y, c=gts, cmap='tab20')
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Classes")
    # scatter = ax.scatter((x_s) , (y_s) , c= (c_s) , cmap='tab10', alpha=1, marker='^')

    # legend1 = ax.legend(*scatter.legend_elements(),
    #                 loc="upper right", title="Classes")
    plt.savefig('1.png')
    plt.close()
    print('done')