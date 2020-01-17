from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV TA\'s tutorial in image classification using pytorch')

    # Datasets parameters
    parser.add_argument('--data_dir_src', type=str, default='../hw3_data/digits', 
                    help="root path to data directory")
    parser.add_argument('--workers', default=0, type=int,
                    help="number of data loading workers (default: 4)")
    
    # training parameters
    parser.add_argument('--gpu', default=0, type=int, 
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--epoch', default=100, type=int,
                    help="num of validation iterations")
    parser.add_argument('--val_epoch', default=1, type=int,
                    help="num of validation iterations")
    parser.add_argument('--train_batch', default=100, type=int,
                    help="train batch size")
    parser.add_argument('--test_batch', default=100, type=int, 
                    help="test batch size")
    parser.add_argument('--lr', default=0.0002, type=float,
                    help="initial learning rate")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                    help="initial learning rate")
    parser.add_argument('--dataset_train', default='m', type=str,
                    help="training dataset")
    parser.add_argument('--dataset_test', default='m', type=str,
                    help="testing dataset")
    parser.add_argument('--adaptation', default='0', type=str,
                    help="testing dataset")
    
    # resume trained model
    parser.add_argument('--resume', type=str, default='', 
                    help="path to the trained model")
    # others
    parser.add_argument('--save_dir', type=str, default='result')
    parser.add_argument('--random_seed', type=int, default=999)

    # Problem 3 arg
    parser.add_argument('--tgt_dir', default='../hw3_data/digits/mnistm/test', type=str,
                    help="target domain dir")

    parser.add_argument('--model', default='mnistm', type=str,
                    help="target domain")

    parser.add_argument('--save_csv', default='test_pred.csv', type=str,
                    help="output file")

    parser.add_argument('--load_model', default='result/model_best_adtaptation_1_src_m_tgt_s.pth.tar', type=str,
                    help="best_model")


    # parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help='Number of filters to use in the generator network')
    parser.add_argument('--ndf', type=int, default=64, help='Number of filters to use in the discriminator network')
    parser.add_argument('--nepochs', type=int, default=100, help='number of epochs to train for')
    # parser.add_argument('--lr', type=float, default=0.0005, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.8, help='beta1 for adam. default=0.5')
    # parser.add_argument('--gpu', type=int, default=1, help='GPU to use, -1 for CPU training')
    parser.add_argument('--outf', default='results', help='folder to output images and model checkpoints')
    parser.add_argument('--method', default='GTA', help='Method to train| GTA, sourceonly')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--adv_weight', type=float, default = 0.1, help='weight for adv loss')
    parser.add_argument('--lrd', type=float, default=0.0001, help='learning rate decay, default=0.0002')
    parser.add_argument('--alpha', type=float, default = 0.3, help='multiplicative factor for target adv. loss')

# DANN/log/model_best_adtaptation_1_src_m_tgt_s.pth.tar


    args = parser.parse_args()

    return args