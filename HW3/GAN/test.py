import os
import torch

import parser
from models import generator 
# import data_test as data

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from PIL import Image


if __name__ == '__main__':
    
    args = parser.arg_parse()

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' prepare data_loader '''
    print('===> prepare noise ...')
    

    nz=torch.load(args.load_noise_1).cuda()

    print('===> prepare model ...')    
    ''' prepare mode '''
    model = generator(args).cuda()

   
    checkpoint = torch.load(args.load_model_1)
    model.load_state_dict(checkpoint)
    img_list=[]
    with torch.no_grad():
    	fake = model(nz).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=8))
    img_only = np.transpose(img_list[-1], (1, 2, 0)).numpy()
    img_only = (img_only *255)
    img_array = np.array(img_only, dtype=np.uint8)
    result = Image.fromarray(img_array)
    result.save(os.path.join(args.save_dir,'fig1_2.jpg'))
    # acc = evaluate(model, test_loader)

    print('First fig created')