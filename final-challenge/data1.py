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
        #self.data_dir = args.data_dir
        self.img_dir = args.data_img_dir

        ''' read the data list '''
        if mode=='gallery':
            csv_path = args.gallery_csv
            self.data = pandas.read_csv(csv_path, header=None)
            self.data = self.data.values.tolist()
        elif mode=='query':
            csv_path = args.query_csv
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
        img_path = self.data[idx][0]
        #tiger_id = np.asarray(self.data[idx][0])
        #tiger_id_index = np.where(self.ordered_unique_IDs == tiger_id)[0][0]
        tiger_name = img_path
        #Read Image
        img_full_path = os.path.join(self.img_dir, img_path)
        tmp_img = Image.open(img_full_path).convert('RGB')
        img = self.transform(tmp_img)
        #Compute local image
        #local = utils.getLocalImage(img_full_path, reshape_shape=self.args.img_shape, patch_size=32)
        #local = self.localTransform(local)

        #show_image_from_tensor(img)
        #print('self.ordered_unique_IDs: {}'.format(self.ordered_unique_IDs))
        #print('tiger_id: {}    tiger_id_index: {}    img_name: {}'.format(tiger_id, tiger_id_index, tiger_name))
        #print('input shape: {}'.format(img.shape))
        
        return img, tiger_name

    def next(self):
        tensors = []
        #tensors_loc = []
        #tiger_id_list = []
        tiger_names_list = []
        if(self.index + self.batch_size < self.len()):
            for i in range(self.batch_size):
                if (i==0):
                    tensors, tiger_name = self.getitem(self.index)
                    tensors = tensors.unsqueeze(dim=0)
                    tiger_names_list.append(tiger_name)
                else:
                    tensor, tiger_name = self.getitem(self.index + i)
                    tensors = torch.cat((tensors, tensor.unsqueeze(dim=0)))
                    tiger_names_list.append(tiger_name)
            self.index = self.index + self.batch_size
        else:
            for i in range(self.len() - self.index):
                if (i==0):
                    tensors, tiger_name = self.getitem(self.index)
                    tensors = tensors.unsqueeze(dim=0)
                    tiger_names_list.append(tiger_name)
                else:
                    tensor, tiger_name = self.getitem(self.index + i)
                    tensors = torch.cat((tensors, tensor.unsqueeze(dim=0)))
                    tiger_names_list.append(tiger_name)
            #print('Length of data after shuffle: {}'.format(self.len()))
            self.index = 0
            self.completed_epochs = self.completed_epochs + 1
        return tensors, tiger_names_list

    
if __name__=="__main__":
    args = parser.arg_parse()
    print("Query")
    data = Data(args, mode="query")
    while(data.completed_epochs < 2):
        data.next()
    print("Gallery")
    data = Data(args, mode="gallery")
    while(data.completed_epochs < 2):
        data.next()