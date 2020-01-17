import os
import json
import torch
import scipy.misc

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image
# from imageio import imread
import numpy as np

MEAN=[0.5, 0.5, 0.5]
STD=[0.5, 0.5, 0.5]

# MEAN=[0.7610319, 0.6311782, 0.56583375]
# STD=[0.17829515729479767, 0.1693871269617932, 0.276091762850216]


class DATA(Dataset):
    def __init__(self, args):

        ''' set up basic parameters for dataset '''
        # self.mode = mode
        self.data_dir = args.data_dir
        self.img_dir = os.path.join(self.data_dir, 'train')
        

        ''' read the data list '''
        d=[]
        for i in os.listdir(self.img_dir):
            # x=[]
            # for j in range(2):
            #     x.append(i)
            d.append(i)
        self.data=d

        ''' set up image path '''
        for d in self.data:
            d = os.path.join(self.img_dir, d)
        
        ''' set up image trainsform '''
        if 1:
            self.transform = transforms.Compose([
                               # transforms.RandomHorizontalFlip(0.5),
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])

        


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        
        ''' get data '''
        img_path = os.path.join(self.img_dir,self.data[idx])
        # seg_path = os.path.join(self.seg_dir,os.listdir(self.seg_dir)[idx])
        
        ''' read image '''
        img = Image.open(img_path).convert('RGB')

        # img = imread(img_path)
        # img1 = np.array(img)
        # img1=np.rollaxis(img, 2)
        # img1 =Image.fromarray(np.rollaxis(img, 2))
        # img1 =Image.fromarray(img.reshape(3,352,448))
        # seg = Image.open(seg_path).convert('L')
        # seg1 = np.array(seg)
        # seg1 = torch.from_numpy(seg1)
        # seg1 = seg1.long()
        # seg1 = np.rollaxis(seg, 2)

        return self.transform(img)