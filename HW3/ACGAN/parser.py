from __future__ import absolute_import
import argparse



def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV TA\'s tutorial in image classification using pytorch')

    # Datasets parameters
    parser.add_argument('--data_dir', type=str, default='../hw3_data/face', 
                    help="root path to data directory")
    # parser.add_argument('--img_dir', type=str, default='hw3_data', 
    #                 help="root path to data directory")
    parser.add_argument('--save_dir', type=str, default='result', 
                    help="root path to data directory")
    parser.add_argument('--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
    
    # training parameters
    parser.add_argument('--gpu', default=0, type=int, 
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--epoch', default=100, type=int,
                    help="num of validation iterations")
    parser.add_argument('--val_epoch', default=1, type=int,
                    help="num of validation iterations")
    parser.add_argument('--train_batch', default=64, type=int,
                    help="train batch size")
    parser.add_argument('--test_batch', default=32, type=int, 
                    help="test batch size")
    parser.add_argument('--lr', default=0.00005, type=float,
                    help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                    help="initial learning rate")
    
    # resume trained model
    parser.add_argument('--resume', type=str, default='', 
                    help="path to the trained model")

    parser.add_argument('--load_model_1', type=str, default='', 
                    help="path to the trained model")

    parser.add_argument('--load_noise_1', type=str, default='', 
                    help="path to the trained model")
    # others
    parser.add_argument('--save_dir1', type=str, default='log')
    parser.add_argument('--random_seed', type=int, default=999)
    parser.add_argument('--nz', type=int, default=100)

    args = parser.parse_args()

    return args