
#python evaluate.py \
#        --dataset ../data.csig/test_hq_no_res_inp \
#        --model w_rl_additive_bleu \
#        --epoch "" \
#        --gpu 0 \
#
#python evaluate.py \
#        --dataset ../data.csig/test_limiao_no_res_inp \
#        --model w_rl_additive_bleu \
#        --epoch "" \
#        --gpu 0 \
#
#python evaluate.py \
#        --dataset ../data.csig/test_xd_no_res_inp \
#        --model w_rl_additive_bleu \
#        --epoch "" \
#        --gpu 0 \

#python evaluate.py \
#        --dataset ../data.csig/test_hq_no_res_inp \
#        --model emnlp/w_gpt_rl/ \
#        --epoch 37 \
#        --gpu 0 \
#
#python evaluate.py \
#        --dataset ../data.csig/test_limiao_no_res_inp \
#        --model emnlp/w_gpt_rl/ \
#        --epoch 80 \
#        --gpu 0 \
#
#python evaluate.py \
#        --dataset ../data.csig/test_xd_no_res_inp \
#        --model emnlp/w_gpt_rl/ \
#        --epoch 37 \
#        --gpu 0 \

for x in 1 4 6
do
    #python evaluate.py \
    #        --dataset ../data.csig/test_limiao_no_res_inp \
    #        --model csig/w_bleu_rl/ \
    #        --epoch $x \
    #        --gpu 0 
    #python evaluate.py \
    #        --dataset ../data.csig/test_hq_no_res_inp \
    #        --model csig/w_bleu_rl/ \
    #        --epoch $x \
    #        --gpu 0 
    python evaluate.py \
            --dataset ../data.csig/test_xd_no_res_inp \
            --model csig/w_bleu_rl/ \
            --epoch $x \
            --gpu 0 
done

