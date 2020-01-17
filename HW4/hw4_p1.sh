# TODO: create shell script for Problem 1
python3 no_rnn/test.py --valid $1 --gt $2 --save_txt $3 --load_model 'no_rnn/log/model_best.pth.tar'
