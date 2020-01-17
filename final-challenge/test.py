import os
import torch
import parser
import numpy as np
from sklearn.metrics import accuracy_score
from model import *
from data import *
from scipy import spatial
import pandas as pd


def evaluate(model, args):
    ''' set model to evaluate mode '''
    gallery_data = Data(args, mode="gallery")
    query_data = Data(args, mode="query")
    model.eval()
    features_gallery = {}

    preds =[]
    gts = []
    # data_list = [['image_name', 'label']]
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        #if (gallery_data.completed_epochs == 0):
        while (gallery_data.completed_epochs < 1):
            imgs, local_imgs , labels, fname= gallery_data.next()

            imgs = imgs.cuda()
            local_imgs = local_imgs.cuda()
            
            if (args.features == 'local'):
                output = model(imgs, local_imgs, 'gallery')
            else:
                output, _ = model(imgs)
            
            pred = output.cpu().numpy()

            for i in range(pred.shape[0]):
                features_gallery[fname[i]]=[pred[i],labels[i]]

        #if (query_data.completed_epochs == 0):

        while (query_data.completed_epochs < 1):
            imgs, local_imgs, labels, fname = query_data.next()
            
            imgs = imgs.cuda()
            local_imgs = local_imgs.cuda()
            
            if (args.features == 'local'):
                output = model(imgs, local_imgs, 'query')
            else:
                output, _ = model(imgs)
            
            pred = output.cpu().numpy()

            for i in range(pred.shape[0]):
                min_dist=np.Inf
                #closest_img=[]
                for name in features_gallery:
                    distance = spatial.distance.cosine(features_gallery[name][0], pred[i])
                    if distance < min_dist:
                        min_dist = distance
                        closest_img = [name, features_gallery[name][1]]
                preds.append(closest_img)
            gt = np.array(labels).squeeze()

            gts.append(gt)


        gts = np.concatenate(gts)
        imgs_names =[]
        imgs_ids = []
        for j in preds:
            imgs_names.append(j[0])
            imgs_ids.append(j[1])
        #imgs_names=preds[:,1:2]
        #imgs_ids = np.concatenate(np.array(preds[:,0:1]))

        return accuracy_score(gts, np.array(imgs_ids)), imgs_names

if __name__=="__main__":
    args = parser.arg_parse()
    if (args.selected_model == 'Resnet18_bn'):
        model = Resnet18_bn()
    elif (args.selected_model == 'Resnet34_bn'):
        model = Resnet34_bn()
    elif (args.selected_model == 'Resnet50_bn'):
        model = Resnet50_bn()
    elif (args.selected_model == 'vgg16'):
        model = vgg16()
    elif (args.selected_model == 'Densenet121'):
        model = Densenet121()      

    model_name = 'best_model_' + args.selected_model + '.pth.tar'
    model.load_state_dict(torch.load(os.path.join(args.save_dir, model_name))) 
    model.cuda()
    summary(model, (3,256,512))
    
    #write predictions
    ans = open(args.output_file,'w')
    acc, preds = evaluate(model, args)
    print ('acc: ', acc)
    for i in range(len(preds)):
        #print('prediction ', preds[i])
        ans.write('{}\n'.format(preds[i]))
    ans.close()

    print('Predictions csv file generated')


