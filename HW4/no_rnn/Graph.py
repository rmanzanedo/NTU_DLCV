import os
import torch

import parser

import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt




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
    # val_loader = torch.utils.data.DataLoader(data.DATA(args, mode='valid'),
    #                                          batch_size=args.train_batch,
    #                                          num_workers=args.workers,
    #                                          shuffle=True)
    train_loader = torch.load('train_resnet.pt')
    train_cls = torch.load('train_cls.pt')
    print(train_loader.shape)
    print('===> T-SNE...')
    X_m = TSNE(n_components=2).fit_transform(train_loader.numpy().squeeze())
    x =np.concatenate(X_m[:,0:1])
    y = np.concatenate(X_m[:,1:2])
    fig, ax = plt.subplots(figsize=(30, 17))
    scatter = ax.scatter(x, y, c=train_cls, cmap='tab20')
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Classes")
    # scatter = ax.scatter((x_s) , (y_s) , c= (c_s) , cmap='tab10', alpha=1, marker='^')

    # legend1 = ax.legend(*scatter.legend_elements(),
    #                 loc="upper right", title="Classes")
    plt.savefig('1.png')
    plt.close()
    print('done')