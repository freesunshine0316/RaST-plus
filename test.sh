#python evaluate.py \
#        --dataset acl \
#        --model emnlp/w_gpt_rl \
#        --epoch 37 \
#        --gpu 0 \

#python evaluate.py \
#        --dataset emnlp \
#        --model w_rl_additive_bleu \
#        --epoch "" \
#        --gpu 0 \

for x in {51..80}
do
    python evaluate.py \
            --dataset emnlp \
            --model acl19/w_gpt_rl \
            --epoch $x \
            --gpu 0 >> log.ori_RaST
done

