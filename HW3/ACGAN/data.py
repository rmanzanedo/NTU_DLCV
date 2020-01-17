import os
import json
import torch
import scipy.misc
import pandas as pd
import csv


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
        self.img_list = sorted(os.listdir(self.img_dir))
        # ''' set up image path '''
        # for j in self.data:
        #     j = os.path.join(self.img_dir, j)
        self.data= [self.img_dir+'/'+photo for photo in self.img_list]

        # self.data=[]
        # self.data1=[]
        # with open(os.path.join(self.data_dir_src, self.src, self.mode+'.csv'), mode='r') as pred:
        #   reader = csv.reader(pred)
        #   {self.data.append(rows[0]) for rows in reader}
        #   # {self.data1.append(rows[1]) for rows in reader}

        # del self.data_src[0]
        # self.data=d
        self.data1=[]

        self.transform = transforms.Compose([
                           # transforms.RandomHorizontalFlip(0.5),
                           transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                           transforms.Normalize(MEAN, STD)
                           ])
        with open(os.path.join(self.data_dir,'train.csv'), mode='r') as pred:
          reader = csv.reader(pred)
          # {self.data.append(rows[0]) for rows in reader}
          {self.data1.append(rows[10]) for rows in reader}
        del self.data1[0] 

        self.data1 = [float(i) for i in self.data1]

        


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        
        ''' get data '''
        img_path = self.data[idx]
        # class_path=os.path.join(self.data_dir,'train.csv')
        # seg_path = os.path.join(self.seg_dir,os.listdir(self.seg_dir)[idx])
        
        ''' read image '''
        img = Image.open(img_path).convert('RGB')

        # img = imread(img_path)
        # img1 = np.array(img)

        # my_csv = pd.read_csv(class_path)
        # column = my_csv.Smiling[idx]
        # labels =np.asarray(column)
        # labels= torch.from_numpy(labels).float()
        labels=self.data1[idx]
        # img1=np.rollaxis(img, 2)
        # img1 =Image.fromarray(np.rollaxis(img, 2))
        # img1 =Image.fromarray(img.reshape(3,352,448))
        # seg = Image.open(seg_path).convert('L')
        # seg1 = np.array(seg)
        # seg1 = torch.from_numpy(seg1)
        # seg1 = seg1.long()
        # seg1 = np.rollaxis(seg, 2)

        return self.transform(img), labels