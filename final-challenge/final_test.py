import os
import torch
import parser
import numpy as np
from sklearn.metrics import accuracy_score
from model1 import *
from data1 import *
from scipy import spatial
import pandas as pd


def evaluate(model, args):
    ''' set model to evaluate mode '''
    gallery_data = Data(args, mode="gallery")
    query_data = Data(args, mode="query")
    model.eval()
    features_gallery = {}

    preds = []
    gts = []
    # data_list = [['image_name', 'label']]
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        # if (gallery_data.completed_epochs == 0):
        while (gallery_data.completed_epochs < 1):
            imgs, fname = gallery_data.next()

            imgs = imgs.cuda()

            output= model(imgs)

            pred = output.cpu().numpy()

            for i in range(pred.shape[0]):
                features_gallery[fname[i]] = [pred[i]]

        # if (query_data.completed_epochs == 0):

        while (query_data.completed_epochs < 1):
            imgs, fname = query_data.next()

            imgs = imgs.cuda()

            output= model(imgs)

            pred = output.cpu().numpy()

            for i in range(pred.shape[0]):
                min_dist = np.Inf
                # closest_img=[]
                for name in features_gallery:
                    distance = spatial.distance.cosine(features_gallery[name][0], pred[i])
                    if distance < min_dist:
                        min_dist = distance
                        closest_img = [name, features_gallery[name][0]]
                preds.append(closest_img)

        imgs_names = []
        for j in preds:
            imgs_names.append(j[0])

        return imgs_names


if __name__ == "__main__":
    args = parser.arg_parse()

    model = Densenet121_global()



    # model_name = 'best_model_' + args.selected_model + '.pth.tar'
    model.load_state_dict(torch.load(args.best_model))
    model.cuda()
    # summary(model, (3, 256, 512))

    # write predictions
    ans = open(args.output_file, 'w')
    print('===>processing')
    preds = evaluate(model, args)
    #ans.write('img_name\n')
    for i in range(len(preds)):
        ans.write('{}\n'.format(preds[i]))
    ans.close()

    print('Predictions csv file generated at {}'.format(args.output_file))