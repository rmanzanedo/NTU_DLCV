import os
import json
import torch
import scipy.misc
import cv2 as cv
import numpy as np
import pandas
import random
from itertools import groupby

import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image

import parser
import utils

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class Data():
    def __init__(self, args, mode='train'):

        self.args = args
        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        self.img_dir = os.path.join(self.data_dir, self.args.data_name, 'imgs')

        ''' read the data list '''
        csv_path = os.path.join(self.data_dir, self.args.data_name, self.mode + '.csv')
        self.data = pandas.read_csv(csv_path, header=None)
        self.data = self.data.values.tolist()
        #print(self.data)
        #print('Number of images in {}.csv: {}'.format(self.mode,len(self.data)))

        ''' set up image transform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([transforms.RandomResizedCrop(size=(256), scale=(random.uniform(0.9,1), random.uniform(0.9,1)), ratio=(2,1), interpolation=2),
                                                 transforms.Resize(size=self.args.img_shape, interpolation=2),
                                                 transforms.RandomHorizontalFlip(p=self.args.flip_p), 
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(MEAN, STD),
                                                 transforms.RandomErasing(p=0.5, scale=(random.uniform(0,0.1), random.uniform(0,0.1)), value=MEAN)])

        elif self.mode == 'gallery' or self.mode == 'query':
            self.transform = transforms.Compose([transforms.Resize(size=self.args.img_shape, interpolation=2),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(MEAN, STD)])

        self.localTransform = transforms.Compose([transforms.ToTensor()])

        #Parameters to be set for data importing in each batch
        self.index = 0
        self.batch_size = self.args.train_batch

        #Order data based on tiger ID
        if (self.mode == 'train'):
            self.data.sort()
        #print(self.data)

        #Get tiger IDs values that will be used in the shuffle
        all_ids = [entry[0] for entry in self.data]
        self.uniqueIDs = np.unique(np.array(all_ids))
        self.ordered_unique_IDs = self.uniqueIDs
        #print('Unique IDs for {} mode are {}'.format(self.mode,self.uniqueIDs))

        #Get number of tigers for each ID
        self.n_imgs_in_index = [len(list(group)) for key, group in groupby(all_ids)]
        #print(np.sum(np.array(self.n_imgs_in_index)))
        #print(self.n_imgs_in_index)
        
        #Get sublists for each tiger
        self.sublists = []
        for i in range(len(self.uniqueIDs)):
            #Finding indeces of all elements of a certain tiger
            indices_to_pick = []
            for k in range(len(self.data)):
                if(self.data[k][0] == self.uniqueIDs[i]):
                    indices_to_pick.append(k)
            #Generating a sublist containing only elements of a certain tiger
            #and append it on a bigger list
            sublist = []
            for k in range(len(indices_to_pick)):
                sublist.append(self.data[indices_to_pick[k]])
            self.sublists.append(sublist)
        #print(self.sublists[10])

        #Set epochs completed number
        self.completed_epochs = 0

        #First shuffle to obtain trainable datas
        #print('Length of data before shuffle: {}'.format(self.len()))
        if(self.mode == 'train'):
            self.customShuffle()
        #print('Length of data after shuffle: {}'.format(self.len()))

    def len(self):
        return len(self.data)

    def getitem(self, idx):

        #Get Data
        img_path = self.data[idx][1]
        tiger_id = np.asarray(self.data[idx][0])
        tiger_id_index = np.where(self.ordered_unique_IDs == tiger_id)[0][0]
        tiger_name = img_path
        #Read Image
        img_full_path = os.path.join(self.img_dir, img_path)
        tmp_img = Image.open(img_full_path).convert('RGB')
        img = self.transform(tmp_img)
        #Compute local image
        local = utils.getLocalImage(img_full_path, reshape_shape=self.args.img_shape, patch_size=32)
        local = self.localTransform(local)

        #show_image_from_tensor(img)
        #print('self.ordered_unique_IDs: {}'.format(self.ordered_unique_IDs))
        #print('tiger_id: {}    tiger_id_index: {}    img_name: {}'.format(tiger_id, tiger_id_index, tiger_name))
        #print('input shape: {}'.format(img.shape))
        
        return img, local, tiger_id, tiger_id_index, tiger_name

    def next(self):
        tensors = []
        tensors_loc = []
        tiger_id_list = []
        tiger_names_list = []
        if(self.index + self.batch_size < self.len()):
            if(self.mode == 'train'):
                printProgressBar(self.index + self.batch_size, self.len(), prefix='Epoch [{}]:'.format(self.completed_epochs + 1))
            for i in range(self.batch_size):
                if (i==0):
                    tensors, tensors_loc, tiger_id, tiger_id_index, tiger_name = self.getitem(self.index)
                    tensors = tensors.unsqueeze(dim=0)
                    tensors_loc = tensors_loc.unsqueeze(dim=0)
                    tiger_id_list.append(tiger_id_index)
                    tiger_names_list.append(tiger_name)
                else:
                    tensor, tensor_loc, tiger_id, tiger_id_index, tiger_name = self.getitem(self.index + i)
                    tensors = torch.cat((tensors, tensor.unsqueeze(dim=0)))
                    tensors_loc = torch.cat((tensors_loc, tensor_loc.unsqueeze(dim=0)))
                    tiger_id_list.append(tiger_id_index)
                    tiger_names_list.append(tiger_name)
            self.index = self.index + self.batch_size
        else:
            '''
            if(self.mode == 'train'):
                printProgressBar(self.len(), self.len(), prefix='Epoch [{}]:'.format(self.completed_epochs + 1))
            for i in range(self.len() - self.index):
                if (i==0):
                    tensors, tiger_id, tiger_id_index, tiger_name = self.getitem(self.index)
                    tensors = tensors.unsqueeze(dim=0)
                    tiger_id_list.append(tiger_id_index)
                    tiger_names_list.append(tiger_name)
                else:
                    tensor, tiger_id, tiger_id_index, tiger_name = self.getitem(self.index + i)
                    tensors = torch.cat((tensors, tensor.unsqueeze(dim=0)))
                    tiger_id_list.append(tiger_id_index)
                    tiger_names_list.append(tiger_name)
            #print('Length of data before shuffle: {}'.format(self.len()))
            '''
            if(self.mode == 'train'):
                printProgressBar(self.len(), self.len(), prefix='Epoch [{}]:'.format(self.completed_epochs + 1))
                self.customShuffle()
            else:
                for i in range(self.len() - self.index):
                    if (i==0):
                        tensors, tensors_loc, tiger_id, tiger_id_index, tiger_name = self.getitem(self.index)
                        tensors = tensors.unsqueeze(dim=0)
                        tensors_loc = tensors_loc.unsqueeze(dim=0)
                        tiger_id_list.append(tiger_id_index)
                        tiger_names_list.append(tiger_name)
                    else:
                        tensor, tensor_loc, tiger_id, tiger_id_index, tiger_name = self.getitem(self.index + i)
                        tensors = torch.cat((tensors, tensor.unsqueeze(dim=0)))
                        tensors_loc = torch.cat((tensors_loc, tensor_loc.unsqueeze(dim=0)))
                        tiger_id_list.append(tiger_id_index)
                        tiger_names_list.append(tiger_name)
            #print('Length of data after shuffle: {}'.format(self.len()))
            self.index = 0
            self.completed_epochs = self.completed_epochs + 1
        return tensors, tensors_loc, tiger_id_list, tiger_names_list

    def customShuffle(self):
        #print('\n\n\n\n\n\n\n\n\n\nShuffle\n\n\n\n\n\n\n\n\n\n')
        new_list = []
        n_imgs_picked_from_index = np.zeros(len(self.uniqueIDs), dtype=int)

        #print('uniqueIDS before shuffle: {}'.format(self.uniqueIDs[0:10]))
        #print('frequencies before shuffle: {}'.format(self.n_imgs_in_index[0:10]))
        c = list(zip(self.uniqueIDs, self.n_imgs_in_index, self.sublists))
        random.shuffle(c)
        self.uniqueIDs, self.n_imgs_in_index, self.sublists = zip(*c)
        #print('uniqueIDS after shuffle: {}'.format(self.uniqueIDs[0:10]))
        #print('frequencies after shuffle: {}'.format(self.n_imgs_in_index[0:10]))
        #print('n_imgs_picked_from_index: {}'.format(n_imgs_picked_from_index))

        #Shuffle all the sublists
        for i in range(len(self.uniqueIDs)):
            random.shuffle(self.sublists[i])

        #print(self.sublists[10])

        n_imgs_with_equal_id_per_batch = int(self.batch_size/self.args.tigers_per_batch)

        while(len(new_list)<self.len()):
            for i in range(len(self.uniqueIDs)):
                if(n_imgs_picked_from_index[i] + n_imgs_with_equal_id_per_batch <= self.n_imgs_in_index[i]):
                    count = 0
                    for k in range(n_imgs_with_equal_id_per_batch):
                        #print('n_imgs_picked_from_index[i]+count: {}'.format(n_imgs_picked_from_index[i]+count))
                        #print('n_imgs_picked_from_index[i]: {}'.format(n_imgs_picked_from_index[i]))
                        #print('count: {}'.format(count))
                        new_list.append(self.sublists[i][n_imgs_picked_from_index[i]])
                        n_imgs_picked_from_index[i] = n_imgs_picked_from_index[i] + 1
                        count += 1
                        if(len(new_list)==self.len()):
                            self.data = new_list
                            return
                else:
                    n_imgs_w_be_picked = self.n_imgs_in_index[i] - n_imgs_picked_from_index[i]
                    for k in range(self.n_imgs_in_index[i] - n_imgs_picked_from_index[i]):
                        new_list.append(self.sublists[i][n_imgs_picked_from_index[i]+k])
                        if(len(new_list)==self.len()):
                            self.data = new_list
                            return
                    n_imgs_picked_from_index[i] = 0
                    count = 0
                    for k in range(n_imgs_with_equal_id_per_batch - n_imgs_w_be_picked):
                        new_list.append(self.sublists[i][n_imgs_picked_from_index[i]])
                        n_imgs_picked_from_index[i] = n_imgs_picked_from_index[i] + 1
                        count += 1
                        if(len(new_list)==self.len()):
                            self.data = new_list
                            return
        self.data = new_list
        #print(self.data)
        #print(self.len())


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '=', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + ' ' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def show_image_from_tensor(tensor):
    import matplotlib.pyplot as plt
    img = tensor.numpy()
    img = np.transpose(img, (1,2,0))
    fig = plt.figure()
    plt.imshow(img)
    plt.show()

if __name__=="__main__":
    args = parser.arg_parse()
    print("Train")
    data = Data(args, mode="train")
    while(data.completed_epochs < 2):
        data.next()
    print("Query")
    data = Data(args, mode="query")
    while(data.completed_epochs < 2):
        data.next()
    print("Gallery")
    data = Data(args, mode="gallery")
    while(data.completed_epochs < 2):
        data.next()