#python train1.py --train_batch=24 --tigers_per_batch=8 --selected_model=rn
#python train1.py --train_batch=16 --tigers_per_batch=4 --selected_model=dn
#python train1.py --train_batch=24 --tigers_per_batch=8 --selected_model=vgg
#python train1.py --train_batch=12 --tigers_per_batch=3 --selected_model=rn-dn
#python train1.py --train_batch=16 --tigers_per_batch=4 --selected_model=dn-rn
#python test1.py --selected_model=rn
#python test1.py --selected_model=dn
#python test1.py --selected_model=vgg
#python test1.py --selected_model=rn-dn
#python test1.py --selected_model=dn-rn
#python final_test.py --data_img_dir data/tigers/imgs --query_csv data/tigers/query1.csv --gallery_csv data/tigers/gallery1.csv --output_file output.csv
bash final.sh data/tigers/imgs data/tigers/query1.csv data/tigers/gallery1.csv output.csv