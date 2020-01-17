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
    def __init__(self, args, mode='train', dataset='m', adaptation='0'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        
        if adaptation=='0' and dataset=='m':
          self.src = 'mnistm'
          self.tgt = None
          self.data_dir_src = args.data_dir_src
          self.data_dir_tgt = None
        elif adaptation=='0' and dataset== 's':
          self.src = 'svhn'
          self.tgt = None
          self.data_dir_src = args.data_dir_src
          self.data_dir_tgt = None
        elif self.mode == 'train' and dataset=='m':
          self.src = 'mnistm'
          self.tgt = 'svhn'
          self.data_dir_src = args.data_dir_src
          self.data_dir_tgt = args.data_dir_src
        elif self.mode == 'train' and dataset=='s':
          self.src = 'svhn'
          self.tgt = 'mnistm'
          self.data_dir_src = args.data_dir_src
          self.data_dir_tgt = args.data_dir_src
        # elif self.mode == 'test' and dataset=='ms':
        #   self.src = 'mnistm'
        #   self.tgt = None
        # elif self.mode == 'test' and dataset=='sm':
        #   self.src = 'svhn'
        #   self.tgt = None
        else:
          print('wrong dataset')
          exit()

        


        self.img_dir_src = os.path.join(self.data_dir_src, self.src, self.mode)

        if self.tgt:
          self.img_dir_tgt = os.path.join(self.data_dir_tgt, self.tgt, self.mode)
        else:
          self.img_dir_tgt = None



        ''' read the data list '''
        self.data_src=[]
        self.data1=[]
        with open(os.path.join(self.data_dir_src, self.src, self.mode+'.csv'), mode='r') as pred:
          reader = csv.reader(pred)
          {self.data_src.append(rows[0]) for rows in reader}
          # {self.data1.append(rows[1]) for rows in reader}

        del self.data_src[0]

        if self.tgt:
          self.data_tgt=[]
          with open(os.path.join(self.data_dir_tgt, self.tgt, self.mode+'.csv'), mode='r') as pred:
            reader = csv.reader(pred)
            {self.data_tgt.append(rows[0]) for rows in reader}
          del self.data_tgt[0]
        else:
          self.data_tgt=None
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
            with open(os.path.join(self.data_dir_src, self.src,'train.csv'), mode='r') as pred:
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
            with open(os.path.join(self.data_dir_src, self.src,'test.csv'), mode='r') as pred:
              reader = csv.reader(pred)
              # {self.data.append(rows[0]) for rows in reader}
              {self.data1.append(rows[1]) for rows in reader}

        # del self.data[0]
            del self.data1[0] 
            self.data1 = [int(i) for i in self.data1]           

    def __len__(self):
        if self.tgt:
          # print(min(len(self.data_src), len(self.data_tgt)))
          # exit()
          return min(len(self.data_src), len(self.data_tgt))
        else:
          return len(self.data_src)

    def __getitem__(self, idx):

        
      if self.tgt:

        ''' get data '''
        src_path = os.path.join(self.img_dir_src, self.data_src[idx])
        tgt_path = os.path.join(self.img_dir_tgt, self.data_tgt[idx])



        
        ''' read image '''
        img_src = Image.open(src_path).convert('RGB')
        img_tgt = Image.open(tgt_path).convert('RGB')
        cls = self.data1[idx]

        # img = zip(self.transform(img_src),self.transform(img_tgt))

        return self.transform(img_src),self.transform(img_tgt), cls

      else:

        src_path = os.path.join(self.img_dir_src, self.data_src[idx])
        
        img_src = Image.open(src_path).convert('RGB')
        cls = self.data1[idx]

        return self.transform(img_src), cls

