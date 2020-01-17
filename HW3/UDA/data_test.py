import os
import json
import torch
import scipy.misc

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

class DATA(Dataset):
    def __init__(self, args, mode='test'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.tgt_dir
        # self.img_dir = os.path.join(self.data_dir, 'imgs')


        ''' read the data list '''
       
        d=[]
        for i in sorted(os.listdir(self.data_dir)):
            d.append(i)
        
        # print(d)
        # exit()
        self.data_tgt=d


         
       

        ''' set up image path '''
        # for d in self.data:
        #     d[0] = os.path.join(self.img_dir, d[0])
        
        ''' set up image trainsform '''
        if  self.mode == 'test':
            self.transform = transforms.Compose([
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])


    def __len__(self):
        return len(self.data_tgt)

    def __getitem__(self, idx):

        ''' get data '''
        img_path = os.path.join(self.data_dir, self.data_tgt[idx])

        # cls = self.data1[idx]
        
        ''' read image '''
        img = Image.open(img_path).convert('RGB')

        return self.transform(img), self.data_tgt[idx]