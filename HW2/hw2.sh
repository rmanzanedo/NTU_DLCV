wget 'https://www.dropbox.com/s/uo110cw6j2j4v0t/base_model.pth.tar?dl=1'
mv base_model.pth.tar?dl=1 base_model.pth.tar
python3 baseline_test.py --resume 'base_model.pth.tar' --img_dir $1 --save_dir $2
