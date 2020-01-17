wget 'https://www.dropbox.com/s/3dp3yl47b3sd3m9/strong_model.pth.tar?dl=1'
mv strong_model.pth.tar?dl=1 strong_model.pth.tar
# RESUME ='strong_model.pth.tar'
python3 strong_test.py --resume 'strong_model.pth.tar' --img_dir $1 --save_dir $2
