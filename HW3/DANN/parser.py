from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV TA\'s tutorial in image classification using pytorch')

    # Datasets parameters
    parser.add_argument('--data_dir_src', type=str, default='../hw3_data/digits', 
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
    parser.add_argument('--train_batch', default=32, type=int,
                    help="train batch size")
    parser.add_argument('--test_batch', default=32, type=int, 
                    help="test batch size")
    parser.add_argument('--lr', default=0.0002, type=float,
                    help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                    help="initial learning rate")
    parser.add_argument('--dataset_train', default='mnistm', type=str,
                    help="training dataset")
    parser.add_argument('--dataset_test', default='mnistm', type=str,
                    help="testing dataset")
    parser.add_argument('--adaptation', default='0', type=str,
                    help="testing dataset")
    
    # resume trained model
    parser.add_argument('--resume', type=str, default='', 
                    help="path to the trained model")
    # others
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--random_seed', type=int, default=999)

    # Problem 3 arg
    parser.add_argument('--tgt_dir', default='../hw3_data/digits/mnistm/test', type=str,
                    help="target domain dir")

    parser.add_argument('--model', default='mnistm', type=str,
                    help="target domain")

    parser.add_argument('--save_csv', default='test_pred.csv', type=str,
                    help="output file")

    parser.add_argument('--load_model', default='log/model_best_adtaptation_1_src_s_tgt_m.pth.tar', type=str,
                    help="best_model")


# DANN/log/model_best_adtaptation_1_src_m_tgt_s.pth.tar


    args = parser.parse_args()

    return args