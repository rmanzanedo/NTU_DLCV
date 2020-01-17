from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV HW2')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='data', 
                    help="root path to data directory")
    parser.add_argument('--data_name', type=str, default='tigers',
                    help="name of the dataset folder")
    parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
    parser.add_argument('--tigers_per_batch', default=8, type=int,
                    help="number of tigers picked in each batch (default: 8)")

    #Data augmentation parameters
    parser.add_argument('--augment_data', type=str, default='True',
                    help = "Set to True if want to augment data, False otherwise")
    parser.add_argument('--flip_p', default=0.5, type=float, 
                    help='flip probability in data augmentation')
    parser.add_argument('--img_shape', default=(256, 256),
                    help='rehape image to this shape during training')
    
    # training parameters
    parser.add_argument('--gpu', default=0, type=int, 
                    help='In homework, please always set to 0')
    parser.add_argument('--epoch', default=100, type=int,
                    help="num of training epochs")
    parser.add_argument('--train_batch', default=32, type=int,
                    help="train batch size")
    # parser.add_argument('--lr', default=1.5e-5, type=float,
    #                 help="initial learning rate")
    parser.add_argument('--lr', default=3.5e-6, type=float,
                        help="initial learning rate")
    parser.add_argument('--saving_name', type=str, default='model.pth.tar', 
                    help="file where is stored the trained model")
    parser.add_argument('--selected_model', default='vgg16', type=str, 
                    help='choose model')
    parser.add_argument('--features', default='global', type=str, 
                    help='global or local')

    # others
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--random_seed', type=int, default=9999)
    parser.add_argument('--output_file', type=str, default='predict.csv')

    parser.add_argument('--data_img_dir', type=str, default='data/tigers/imgs')
    parser.add_argument('--gallery_csv', type=str, default='data/tigers/gallery.csv')
    parser.add_argument('--query_csv', type=str, default='data/tigers/query.csv')
    parser.add_argument('--best_model', type=str, default='best_model.pth.tar')

    args = parser.parse_args()

    return args
