#python evaluate.py \
#        --dataset emnlp \
#        --model acl/w_gpt_rl \
#        --epoch $1 \
#        --gpu 0 \

python evaluate.py \
        --dataset coai \
        --model coai/v2_roberta_large_52_8944 \
        --epoch "" \
        --gpu 3 \
        --span_thres 0.5 \
        --bert_path hfl/chinese-roberta-wwm-ext-large 
