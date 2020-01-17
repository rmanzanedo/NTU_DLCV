if [[ $2 == 'mnistm' ]]
then
model='DANN/model_best_adtaptation_1_src_s_tgt_m.pth.tar'
echo "target mnistm"
elif [[ $2 == 'svhn' ]]
then
model='DANN/model_best_adtaptation_1_src_m_tgt_s.pth.tar'
echo "target svhn"
else
echo "$2 wrong"
fi

python DANN/test.py --tgt_dir $1 --model $2 --save_csv $3 --load_model $model