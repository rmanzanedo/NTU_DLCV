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
    

    test_noise=torch.load(args.load_noise_1).cuda()

    print('===> prepare model ...')    
    ''' prepare mode '''
    gen = generator().cuda()

    img_list=[]
    sml_out = torch.ones(10).cuda()
    no_sml_out = torch.zeros(10).cuda()
    test_noise_sml = torch.cat((test_noise,sml_out.view(10,1,1,1)),1)
    test_noise_no_sml = torch.cat((test_noise, no_sml_out.view(10,1,1,1)),1)
    with torch.no_grad():
        fake = gen(test_noise_sml).detach().cpu()
    # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    
    with torch.no_grad():
        fake_no = gen(test_noise_no_sml).detach().cpu()
    total_fake=torch.cat((fake.view(10,3,64,64), fake_no.view(10,3,64,64)),0)
    # print(total_fake.shape)
    # exit()
    img_list.append(vutils.make_grid(total_fake, padding=2, normalize=True, nrow=10))
    img_only = np.transpose(img_list[-1], (1, 2, 0)).numpy()
    img_only = (img_only *255)
    img_array = np.array(img_only, dtype=np.uint8)
    result = Image.fromarray(img_array)
    result.save(os.path.join(args.save_dir,'fig2_2.jpg'))
    # acc = evaluate(model, test_loader)

    print('Second fig created')
    print('done')