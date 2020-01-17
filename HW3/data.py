import os
import json
import torch
import scipy.misc
import csv

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

class DATA(Dataset):
    def __init__(self, args, mode='train', dataset='m'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        if dataset=='m':
          self.dataset = 'mnistm'
        elif dataset== 's':
          self.dataset = 'svhn'
        else:
          print('wrong dataset')
          exit()
        self.img_dir = os.path.join(self.data_dir, self.dataset, self.mode)

        ''' read the data list '''
        self.data=[]
        self.data1=[]
        with open(os.path.join(self.data_dir, self.dataset, self.mode+'.csv'), mode='r') as pred:
          reader = csv.reader(pred)
          {self.data.append(rows[0]) for rows in reader}
          # {self.data1.append(rows[1]) for rows in reader}

        del self.data[0]
        # del self.data1[0]

        ''' set up image path '''
        # for d in self.data:
        #     d[0] = os.path.join(self.img_dir, d[0])
        
        ''' set up image trainsform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([
                               # transforms.RandomHorizontalFlip(0.5),
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])
            with open(os.path.join(self.data_dir, self.dataset,'train.csv'), mode='r') as pred:
              reader = csv.reader(pred)
              # {self.data.append(rows[0]) for rows in reader}
              {self.data1.append(rows[1]) for rows in reader}

        # del self.data[0]
            del self.data1[0] 

            self.data1 = [int(i) for i in self.data1]
            # for i in range(len(data1)):
            #   self.data1[i]=()

        elif self.mode == 'test':
            self.transform = transforms.Compose([
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])
            with open(os.path.join(self.data_dir, self.dataset,'test.csv'), mode='r') as pred:
              reader = csv.reader(pred)
              # {self.data.append(rows[0]) for rows in reader}
              {self.data1.append(rows[1]) for rows in reader}

        # del self.data[0]
            del self.data1[0] 
            self.data1 = [int(i) for i in self.data1]           

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ''' get data '''
        img_path = os.path.join(self.img_dir, self.data[idx])


        
        ''' read image '''
        img = Image.open(img_path).convert('RGB')
        cls = self.data1[idx]

        return self.transform(img), torch.tensor(cls)