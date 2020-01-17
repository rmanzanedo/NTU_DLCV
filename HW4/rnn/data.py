import csv
import reader
import os
import torch
import scipy.misc
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


class DATA(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        self.vid_dir = os.path.join(self.data_dir, 'video', self.mode)
        self.label_dir= os.path.join(self.data_dir, 'label')


        ''' read the data list '''
        self.label_path = os.path.join(self.label_dir, 'gt_'+ self.mode + '.csv')
        self.data = reader.getVideoList(self.label_path)

        ''' set up image path '''
        # print(self.data['Video_category'], len(self.data))
        # exit()

        ''' set up image trainsform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                # transforms.Normalize(MEAN, STD)
            ])

        elif self.mode == 'valid' or self.mode == 'test':
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                # transforms.Normalize(MEAN, STD)
            ])

    def __len__(self):
        return len(self.data['Video_name'])

    def __getitem__(self, idx):
        # print(idx)
        ''' get data '''
        frames = reader.readShortVideo(self.vid_dir, self.data['Video_category'][idx],self.data['Video_name'][idx])
        # print(frames.shape)
        # frames = Image.fromarray(frames)

        label = int(self.data['Action_labels'][idx])
        ''' read image '''


        # return torch.from_numpy(frames),label
        return frames ,label