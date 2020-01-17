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
import matplotlib.animation as animation
# from matplotlib import pyplot as plt
import os

import data
import parser
# import models
import discriminator as d
import generator as g
from PIL import Image




def make_some_noise():
    return torch.rand(args.train_batch,100,1,1)

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path) 

# defining generator class


# def log_images(images, num_images, epoch, n_batch, num_batches, format='NCHW', normalize=True):
#     '''
#     input images are expected in format (NCHW)
#     '''
#     if type(images) == np.ndarray:
#         images = torch.from_numpy(images)
        
#     if format=='NHWC':
#         images = images.transpose(1,3)
        

#     # step = Logger._step(epoch, n_batch, num_batches)
#     # img_name = '/images{}'.format(epoch)

#     # Make horizontal grid from image tensor
#     horizontal_grid = vutils.make_grid(
#         images, normalize=normalize, scale_each=True)
#     # Make vertical grid from image tensor
#     nrows = int(np.sqrt(num_images))
#     grid = vutils.make_grid(
#         images, nrow=nrows, normalize=True, scale_each=True)

#     # Add horizontal images to tensorboard
#     # self.writer.add_image(img_name, horizontal_grid, step)

#     # Save plots
#     save_torch_images(horizontal_grid, grid, epoch, n_batch)


# def save_torch_images( horizontal_grid, grid, epoch, n_batch, plot_horizontal=True):
#     out_dir = './result'
#     # Logger._make_dir(out_dir)

#     # Plot and save horizontal
#     # fig = plt.figure(figsize=(16, 16))
#     # plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
#     # plt.axis('off')
#     # if plot_horizontal:
#     #     display.display(plt.gcf())
#     # plt.show()
#     # # _save_images(fig, epoch, n_batch, 'hori')
#     # plt.close()

#         # Save squared
#     fig = plt.figure()
#     plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
#     plt.axis('off')
#     plt.savefig('result/img/_epoch_{}_batch_{}.png'.format(str(epoch), str(n_batch)))
#     # plt.show()
#     # _save_images(fig, epoch, n_batch)
#     plt.close()

# def _save_images(self, fig, epoch, n_batch, comment=''):
#     out_dir = './result'
#     # Logger._make_dir(out_dir)
#     fig.savefig('{}/_epoch_{}_batch_{}.png'.format(out_dir,str(epoch), str(n_batch)))


        
        
#     print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
#         epoch,num_epochs, n_batch, num_batches)
#         )
#     print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
#     print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))

# def vectors_to_images(vectors):
#     return vectors.view(vectors.size(0), 3, 64, 64)


# def plot_img(array,number=None):
#     array = array.detach()
#     array = array.reshape(64,64,3)
    
#     plt.imshow(array,cmap='binary')
#     plt.xticks([])
#     plt.yticks([])
#     if number:
#         plt.xlabel(number,fontsize='x-large')
#     plt.show()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
# d_steps = 100
# g_steps = 100
if __name__=='__main__':

    args = parser.arg_parse()

    torch.cuda.set_device(args.gpu)
        
    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    # torch.cuda.manual_seed(args.random_seed)

    # data_transforms = transforms.Compose([transforms.ToTensor()])
    # mnist_trainset = datasets.MNIST(root='./data', train=True,    
    #                            download=True, transform=data_transforms)


    # # batch_size=500
    # train_loader = t_data.DataLoader(mnist_trainset, 
    #                                            batch_size=args.train_batch,
    #                                            shuffle=True
    #                                            )

    print('===> prepare dataloader ...')
    # train_loader = torch.utils.data.DataLoader(data.DATA(args),
    #                                                batch_size=args.train_batch, 
    #                                                num_workers=args.workers,
    #                                                shuffle=True)

    dataset = datasets.ImageFolder(root= args.data_dir,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch,
                                             shuffle=True, num_workers=args.workers)



    print('===> prepare model ...')
    # dis = models.discriminator(3*64*64,1).cuda()

    # gen = models.generator(100,3*64*64).cuda()

    # dis = models.discriminator(args.train_batch).cuda()
    # gen = models.generator(args.train_batch).cuda()

    dis = d.discriminator(args.train_batch).cuda()
    gen = g.generator(args.train_batch).cuda()

    # dis.apply(weights_init)
    # gen.apply(weights_init)
    # print(gen)
    # print(dis)
    # exit() 

    criteriond1 = nn.BCELoss()
    optimizerd1 = optim.Adam(dis.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # criteriond2 = nn.BCELoss()
    optimizerd2 = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))

    writer = SummaryWriter(os.path.join(args.save_dir1, 'train_info'))

    # printing_steps = 200

    # epochs = 1

    print('===> start training ...')
    iters = 0
    best_acc = 0
  
    for epoch in range(1, args.epoch+1):
        ave_real=0
        ave_fake=0
        ave_fake_2=0
        
       # dis.train()
        if (epoch ) == 90:
            optimizerd1.param_groups[0]['lr'] /= 2
            optimizerd2.param_groups[0]['lr'] /= 2
            print("learning rate change!")
        if (epoch + 1) == 120:
            optimizerd1.param_groups[0]['lr'] /= 2
            optimizerd2.param_groups[0]['lr'] /= 2
            print("learning rate change!")
        

        if (epoch + 1) == 160:
            optimizerd1.param_groups[0]['lr'] /= 2
            optimizerd2.param_groups[0]['lr'] /= 2
            print("learning rate change!")

        if (epoch + 1) == 200:
            optimizerd1.param_groups[0]['lr'] /= 2
            optimizerd2.param_groups[0]['lr'] /= 2
            print("learning rate change!")
        

        for idx, data in enumerate(train_loader, 0):

            train_info = 'Epoch: [{0}][{3}][{1}/{2}]'.format(epoch, idx+1, len(train_loader), 'dis')
            iters += 1
            dis.zero_grad()
        
            # imgs = imgs.cuda()
            # inp_real_x = imgs.reshape(args.train_batch,3,64,64).cuda()
            imgs=data[0]
            b_size=imgs.size(0)
            dis_real_out = dis(imgs.cuda()).view(-1)
            # print(args.train_batch, dis_real_out.shape)
            dis_real_loss = criteriond1(dis_real_out, Variable(torch.ones(b_size).cuda()))
            # label_real = torch.full((b_size,), 1).cuda()
            # dis_real_loss =criteriond1 (dis_real_out, label_real)
            dis_real_loss.backward()

            # if 1:
            # if idx%2!=0: 

            inp_fake_x_gen = make_some_noise().cuda()        #output from generator is generated        
            dis_inp_fake_x = gen(inp_fake_x_gen).detach()
            # dis_inp_fake_x = gen(torch.rand(b_size,100,1,1).cuda()).detach()
            dis_fake_out = dis(dis_inp_fake_x).view(-1)
            dis_fake_loss = criteriond1(dis_fake_out, Variable(torch.zeros(args.train_batch).cuda()))
            
            # label_fake = torch.full((b_size,), 0).cuda()
            # dis_fake_loss =criteriond1 (dis_fake_out, label_fake)
            dis_loss = dis_real_loss + dis_fake_loss
           # optimizerd1.zero_grad()
            
            dis_fake_loss.backward()
            optimizerd1.step()     
            print(train_info, end= '\r')
            ave_real+= dis_real_out.mean().item()
            ave_fake+= dis_fake_out.mean().item()

            
            # print(dis_real_loss.data.cpu().numpy(), dis_fake_loss.data.cpu().numpy(), dis_real_loss.data.cpu().numpy()+dis_fake_loss.data.cpu().numpy())

            # train_info = 'Epoch: [{0}][{3}][{1}/{2}]'.format(epoch, idx+1, len(train_loader), 'gen')
            iters += 1
            gen.zero_grad()

            #generating data for input for generator
            gen_inp = make_some_noise().cuda()
        
            gen_out = gen(inp_fake_x_gen)
            # gen_out = gen(torch.rand(b_size,100,1,1).cuda())
            dis_out_gen_training = dis(gen_out).view(-1)
            gen_loss = criteriond1(dis_out_gen_training, Variable(torch.ones(args.train_batch)).cuda())


            # gen_loss =criteriond1 (dis_out_gen_training, label_fake)
            
            #optimizerd2.zero_grad()
            gen_loss.backward()
            ave_fake_2 += dis_out_gen_training.mean().item()
            optimizerd2.step()


            if (idx+1)%250 ==0:    
                print(train_info)
                print('Average Real images: ', ave_real/250 )
                print('Average Fake images: ', ave_fake/250 )
                print('Average Fake images 2: ', ave_fake_2/250 )
                ave_real=0
                ave_fake=0
                ave_fake_2=0
            # print(train_info)
            # print(gen_loss.data.cpu().numpy())
            # print(dis_real_loss.cpu(), dis_fake_loss.cpu(), (dis_fake_loss+dis_real_loss).cpu())
            
            # writer.add_scalar('loss_real', dis_real_loss.data.cpu().numpy(), iters)
            # train_info_real += ' loss_real: {:.4f}'.format(dis_real_loss.data.cpu().numpy())

            # writer.add_scalar('loss_fake', dis_fake_loss.data.cpu().numpy(), iters)
            # train_info_fake += ' loss_fake: {:.4f}'.format(dis_fake_loss.data.cpu().numpy())

            # writer.add_scalar('loss_dis', dis_real_loss.data.cpu().numpy()+dis_fake_loss.data.cpu().numpy(), iters)
            # train_info_dis += ' loss_dis: {:.4f}'.format(dis_real_loss.data.cpu().numpy()+dis_fake_loss.data.cpu().numpy())
                

        print(train_info)
        # iters = 0
        # #gen.train()
        

        # for idx in range(int(40000/args.train_batch)):

        #     train_info = 'Epoch: [{0}][{3}][{1}/{2}]'.format(epoch, idx+1, len(train_loader), 'gen')
        #     iters += 1
        #     optimizerd2.zero_grad()

        #     #generating data for input for generator
        #     gen_inp = make_some_noise().cuda()
            
        #     gen_out = gen(gen_inp)
        #     dis_out_gen_training = dis(gen_out)
        #     gen_loss = criteriond2(dis_out_gen_training, Variable(torch.ones(args.train_batch,1)).cuda())
            
        #     #optimizerd2.zero_grad()
        #     gen_loss.backward()
        #     optimizerd2.step()

        #     print(train_info)
        #     print(gen_loss.data.cpu().numpy())

            # writer.add_scalar('loss_gen', gen_loss.data.cpu().numpy(), iters)
            # train_info_dis += ' loss_gen: {:.4f}'.format(gen_loss.data.cpu().numpy())

        #if epoch%args.val_epoch==0:
         #   plot_img(gen_out[0].cpu())
          #  plot_img(gen_out[1].cpu())
           # plot_img(gen_out[2].cpu())
            #plot_img(gen_out[3].cpu())
            #print("\n\n")
        if (epoch) % args.val_epoch  == 0: 
            # pics_in_grid=32
            # test_noise=torch.rand(pics_in_grid,100,1,1).cuda()
            # test_images = vectors_to_images(gen(test_noise).cpu())
            # test_images = test_images.data            
            # log_images(
            #     test_images, pics_in_grid, 
            #     epoch, epoch, args.train_batch
            # );
            img_list=[]
            test_noise=torch.rand(32,100,1,1).cuda()
            with torch.no_grad():
                fake = gen(test_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            fig = plt.figure(figsize=(4,8))
            
            plt.axis("off")
            ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
            # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
            # plt.show()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=8))
            img_only = np.transpose(img_list[-1], (1, 2, 0)).numpy()
            img_only = (img_only *255)
            img_array = np.array(img_only, dtype=np.uint8)
            result = Image.fromarray(img_array)
            result.save('result/img/_epoch_{0}_batch_{1}.png'.format(epoch, args.train_batch))







        save_model(dis, os.path.join(args.save_dir,'dis' , 'dis_{}.pth.tar'.format(epoch)))
        save_model(gen, os.path.join(args.save_dir,'gen' , 'gen_{}.pth.tar'.format(epoch)))
        torch.save(test_noise.cpu(),os.path.join(args.save_dir,'noise' , 'nos_{}.pth.tar'.format(epoch)))


        # print (epoch)


        # # training discriminator
        # for d_step in range(d_steps):
        #     dis.zero_grad()
            
        #     # training discriminator on real data
        #     # print(train_loader[0])
            
        #     # for inp_real in train_loader:
        #     #     inp_real_x = inp_real
        #     #     print(inp_real_x.size())
        #     #     break

            
            
            

        #     # training discriminator on data produced by generator
            
            

            

            
            
            
                
        # # training generator
        # for g_step in range(g_steps):
        #     gen.zero_grad()
            
            

        #     print(gen_loss.cpu())

            
        