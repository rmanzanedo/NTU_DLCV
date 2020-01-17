# python DANN/DANN.py --dataset_train 'm' --dataset_test 's' --epoch 50 --adaptation '1' --data_dir_src hw3_data/digits --save_dir DANN/log
# python DANN/DANN.py --dataset_train 's' --dataset_test 'm' --epoch 50 --adaptation '1' --data_dir_src hw3_data/digits --save_dir DANN/log
bash hw3_p3.sh hw3_data/digits/mnistm/test mnistm test_pred.csv
python hw3_eval.py test_pred.csv hw3_data/digits/mnistm/test.csv
bash hw3_p3.sh hw3_data/digits/svhn/test svhn test_pred.csv
python hw3_eval.py test_pred.csv hw3_data/digits/svhn/test.csv
bash hw3_p4.sh hw3_data/digits/mnistm/test mnistm test_pred.csv
python hw3_eval.py test_pred.csv hw3_data/digits/mnistm/test.csv
bash hw3_p4.sh hw3_data/digits/svhn/test svhn test_pred.csv
python hw3_eval.py test_pred.csv hw3_data/digits/svhn/test.csv