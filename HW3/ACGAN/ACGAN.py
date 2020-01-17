import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data as t_data
import torchvision.datasets as datasets
from torchvision import transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
from PIL import Image
# from matplotlib import pyplot as plt
import os

import data
import parser
import models




def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)

def make_some_noise(b_size):
    return torch.rand(b_size,args.nz,1,1)

if __name__=='__main__':

    args = parser.arg_parse()

    torch.cuda.set_device(args.gpu)
        
    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    # data_transforms = transforms.Compose([transforms.ToTensor()])
    # mnist_trainset = datasets.MNIST(root='./data', train=True,    
    #                            download=True, transform=data_transforms)


    # # batch_size=500
    # train_loader = t_data.DataLoader(mnist_trainset, 
    #                                            batch_size=args.train_batch,
    #                                            shuffle=True
    #                                            )

    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data.DATA(args),
                                                   batch_size=args.train_batch, 
                                                   num_workers=args.workers,
                                                   shuffle=True)

    print('===> prepare model ...')
    # dis = models.discriminator(3*64*64,1).cuda()

    # gen = models.generator(100,3*64*64).cuda()

    dis = models.discriminator().cuda()
    gen = models.generator().cuda()

    criteriond1 = nn.BCELoss()
    optimizerd1 = optim.Adam(dis.parameters(), lr=0.0002, betas=(0.5, 0.999))
 
    # criteriond3 = nn.NLLLoss().cuda()
    # optimizerd1 = optim.Adam(dis.parameters(), lr=0.0002, betas=(0.5, 0.999))
    

    optimizerd2 = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # writer = SummaryWriter(os.path.join(args.save_dir1, 'train_info'))

    # printing_steps = 200
    fixed_noise = torch.randn(64, args.nz, 1, 1).cuda()
    real_label = 1
    fake_label = 0

    # epochs = 1

    print('===> start training ...')
    iters = 0
    best_acc = 0
    for epoch in range(1, args.epoch+1):

        ave_real=0
        ave_fake=0
        ave_fake_2=0
        avg_real = 0
        avg_fake = 0


        for idx, (imgs, sml) in enumerate(train_loader):

            # print(type(sml[0]),sml[0])
            # exit()
            dis.zero_grad()
            train_info = 'Epoch: [{0}][{3}][{1}/{2}]'.format(epoch, idx+1, len(train_loader), 'dis')
            iters += 1
            
            # print(imgs)
            # exit()
            
            sml = sml.float().cuda()
            imgs = imgs.cuda()
            batch_size = imgs.size(0)
            label = torch.full((batch_size,), real_label).cuda()
            # print(sml)
            

            # TRAIN DISCRIMINATOR
            
            dis_real_out, dis_real_sml = dis(imgs)
            # dis_real_loss = criteriond1(dis_real_out, Variable(torch.ones(batch_size).cuda()))
            dis_real_loss = criteriond1(dis_real_out, label)
            # print(dis_real_sml.view(-1), sml)
            # exit()
            dis_real_loss_sml = criteriond1(dis_real_sml, sml)
            dis_real = dis_real_loss + dis_real_loss_sml
            dis_real.backward()


            inp_fake_x_gen = torch.randn(batch_size, 100, 1, 1).cuda()
            fake_sml = torch.randint(0,2,size=(batch_size,1)).float().cuda()
            # exit()
            noise = torch.cat((inp_fake_x_gen,fake_sml.view(batch_size, 1, 1, 1)), 1)
            dis_inp_fake_x = gen(noise)
            label.fill_(fake_label)
            dis_fake_out, dis_fake_sml = dis(dis_inp_fake_x.detach())
            # dis_fake_loss = criteriond1(dis_fake_out, Variable(torch.zeros(batch_size).cuda()))
            dis_fake_loss = criteriond1(dis_fake_out,label)
            dis_fake_loss_sml = criteriond1(dis_fake_sml, fake_sml)
            dis_fake = (dis_fake_loss + dis_fake_loss_sml)
            dis_fake.backward()
            # print(train_info, end='\r')
            #print(dis_real_loss.data.cpu().numpy(), dis_fake_loss.data.cpu().numpy(), dis_real_loss.data.cpu().numpy()+dis_fake_loss.data.cpu().numpy())

            optimizerd1.step()
            ave_real+= dis_real.mean().item()
            ave_fake+= dis_fake.mean().item()  
            avg_real+= dis_real_out.mean().item()
            avg_fake+= dis_fake_out.mean().item()  

        # iters = 0
        #gen.train()
            # TRAIN GENERATOR
            train_info = 'Epoch: [{0}][{3}][{1}/{2}]'.format(epoch, idx+1, len(train_loader), 'gen')

            gen.zero_grad()
            label.fill_(real_label)
            # inp_fake_x_gen = make_some_noise(batch_size*2).cuda()

            # fake_sml = torch.randint(0,2,size=(batch_size*2,1)).float().cuda()
            # noise = torch.cat( (inp_fake_x_gen,fake_sml.view(batch_size*2, 1, 1, 1)), 1)
            # dis_inp_fake_x = gen(noise)#.detach()
            #generating data for input for generator
            # dis_out_gen_training, smile_fake = dis(dis_inp_fake_x)
            dis_out_gen_training, smile_fake = dis(dis_inp_fake_x)
            # gen_loss = criteriond1(dis_out_gen_training, torch.ones(batch_size*2).cuda())
            gen_loss = criteriond1(dis_out_gen_training, label)
            dis_fake_loss_sml = criteriond1(smile_fake, fake_sml)
            g_loss = (gen_loss + dis_fake_loss_sml)
            
            #optimizerd2.zero_grad()
            g_loss.backward()
            optimizerd2.step()
            ave_fake_2 += g_loss.mean().item()

        # for idx in range(int(40000/args.train_batch)):
            

            print(train_info, end='\r')
            # print(gen_loss.data.cpu().numpy())

            # writer.add_scalar('loss_gen', gen_loss.data.cpu().numpy(), iters)
            # train_info_dis += ' loss_gen: {:.4f}'.format(gen_loss.data.cpu().numpy())

        #if epoch%args.val_epoch==0:
         #   plot_img(gen_out[0].cpu())
          #  plot_img(gen_out[1].cpu())
           # plot_img(gen_out[2].cpu())
            #plot_img(gen_out[3].cpu())
            #print("\n\n")
            if (idx+1)%100 ==0:    
                    print(train_info)
                    print('Average Real images: ', ave_real/100 )
                    print('Average Fake images: ', ave_fake/100 )
                    print('Average Fake images 2: ', ave_fake_2/100 )
                    print('Average real value: ', avg_real/100 )
                    print('Average Fake value: ', avg_fake/100 )
                    ave_real=0
                    ave_fake=0
                    ave_fake_2=0
                    avg_real = 0
                    avg_fake = 0


        if (epoch) % args.val_epoch  == 0: 
            img_list=[]
            test_noise=torch.rand(10,100,1,1).cuda()
            sml_out = torch.full((10,), real_label).cuda()
            no_sml_out = torch.full((10,), fake_label).cuda()
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
            result.save('result/img/_epoch_{0}_batch_{1}.png'.format(epoch, args.train_batch))
            # fig = plt.figure(figsize=(10,2))
            # plt.axis("off")
            # ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]       # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
            # # plt.show()
            # img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=10))
            # img_only = np.transpose(img_list[-1], (1, 2, 0)).numpy()
            # img_only = (img_only *255)
            # img_array = np.array(img_only, dtype=np.uint8)
            # result = Image.fromarray(img_array)
            # result.save('result/img/_epoch_{0}_batch_{1}.png'.format(epoch, args.train_batch))
        save_model(dis, os.path.join(args.save_dir,'dis' , 'dis_{}.pth.tar'.format(epoch)))
        save_model(gen, os.path.join(args.save_dir,'gen' , 'gen_{}.pth.tar'.format(epoch)))
        torch.save(test_noise.cpu(),os.path.join(args.save_dir,'noise' , 'nos_{}.pth.tar'.format(epoch)))

        # result.save('result/img/_epoch_{0}_batch_{1}.png'.format(epoch, args.train_batch))
        # if (epoch) % args.val_epoch  == 0: 
        #     pics_in_grid=32
        #     test_noise=torch.rand(pics_in_grid,100,1,1).cuda()
        #     test_images = vectors_to_images(gen(test_noise).cpu())
        #     test_images = test_images.data            
        #     log_images(
        #         test_images, pics_in_grid, 
        #         epoch, epoch, args.train_batch
        #     );
           
            
        