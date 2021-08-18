#python train.py \
#        --dataset coai \
#        --lower_case \
#        --fold 10 \
#        --model coai_p1p2/fold_10 \
#        --gpu 6 \
#        --restore_point experiments/coai_p1p2/pretrained_checkpoints/fold10.pt \
#        --bert_path hfl/chinese-roberta-wwm-ext-large 

python train.py \
        --dataset coai \
        --lower_case \
        --fold 8 \
        --model coai_p1p2/fold_8 \
        --gpu 5 \
        --restore_point experiments/coai_p1p2/pretrained_checkpoints/fold8.pt \
        --bert_path hfl/chinese-roberta-wwm-ext-large 

