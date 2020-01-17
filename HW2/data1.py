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

MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]

class DATA(Dataset):
    def __init__(self, args):

        ''' set up basic parameters for dataset '''
        self.mode = 'val'
        self.data_dir = args.img_dir
        self.img_dir = args.img_dir
#        self.seg_dir = os.path.join(self.data_dir,mode, 'seg')

        ''' read the data list '''
        d=[]
        for i in os.listdir(self.img_dir):
            x=[]
            for j in range(2):
                x.append(i)
            d.append(x)
        self.data=d

        ''' set up image path '''
#        for d in self.data:
#            d[0] = os.path.join(self.img_dir, d[0])
        
        ''' set up image trainsform '''
#        if self.mode == 'train':
#            self.transform = transforms.Compose([
#                               # transforms.RandomHorizontalFlip(0.5),
#                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
#                               transforms.Normalize(MEAN, STD)
#                               ])
#
#        elif self.mode == 'val' or self.mode == 'test':
        self.transform = transforms.Compose([
                            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                            transforms.Normalize(MEAN, STD)
                            ])

#
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        
        ''' get data '''
        img_path = os.path.join(self.img_dir,os.listdir(self.img_dir)[idx])
#        seg_path = os.path.join(self.seg_dir,os.listdir(self.seg_dir)[idx])
        
        ''' read image '''
        img = Image.open(img_path).convert('RGB')

        # img = imread(img_path)
        img1 = np.array(img)
        # img1=np.rollaxis(img, 2)
        # img1 =Image.fromarray(np.rollaxis(img, 2))
        # img1 =Image.fromarray(img.reshape(3,352,448))
#        seg = Image.open(seg_path).convert('L')
#        seg1 = np.array(seg)
#        seg1 = torch.from_numpy(seg1)
#        seg1 = seg1.long()
        # seg1 = np.rollaxis(seg, 2)

        return self.transform(img1), os.listdir(self.img_dir)[idx]