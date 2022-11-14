#python evaluate.py \
#        --dataset emnlp \
#        --model acl/w_gpt_rl \
#        --epoch $1 \
#        --gpu 0 \

python evaluate.py \
        --dataset coai \
        --subset dev_neg \
        --model coai/every_1/66 \
        --epoch "" \
        --gpu 0 \
        --span_thres 0.5 \
        --bert_path hfl/chinese-roberta-wwm-ext-large 
