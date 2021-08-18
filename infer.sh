
#for x in fold_10_61_8876 fold_1_60_8897 fold_2_73_8945 fold_3_47_8945 fold_4_73_8923 fold_5_71_8822 fold_6_75_8906 fold_7_80_8846 fold_8_28_8877 fold_9_80_8925
#do
#    python evaluate.py \
#            --dataset ./data/coai.test_with_noise.sentences.txt \
#            --model coai/$x \
#            --epoch "" \
#            --gpu 6 \
#            --span_thres 0.5 \
#            --bert_path hfl/chinese-roberta-wwm-ext-large \
#            --dump_decisions_instead
#done

for x in fold_10_27_8913   fold_1_62_8942   fold_2_80_8932  fold_3_36_8911  fold_4_60_8936  fold_5_71_8849  fold_6_80_8943  fold_7_62_8952  fold_8_60_8902  fold_9_23_8970
do
    python evaluate.py \
            --dataset ./data/coai.test_with_noise.sentences.txt \
            --model coai_onlyp2/$x \
            --epoch "" \
            --gpu 7 \
            --span_thres 0.5 \
            --bert_path hfl/chinese-roberta-wwm-ext-large \
            --dump_decisions_instead
done

#for x in fold_10_80_8947  fold_1_57_8966  fold_2_76_8889  fold_3_77_8931  fold_4_77_8994  fold_5_43_8855  fold_6_51_8968  fold_7_34_8912  fold_8_79_8934
#do
#    python evaluate.py \
#            --dataset ./data/coai.test_with_noise.sentences.txt \
#            --model coai_p1p2/$x\_extlen \
#            --epoch "" \
#            --gpu 7 \
#            --span_thres 0.5 \
#            --bert_path hfl/chinese-roberta-wwm-ext-large \
#            --dump_decisions_instead
#done

#for x in fold_9_79_9007
#do
#    python evaluate.py \
#            --dataset ./data/coai.test_with_noise.sentences.txt \
#            --model coai_p1p2/$x \
#            --epoch "" \
#            --gpu 7 \
#            --span_thres 0.5 \
#            --bert_path hfl/chinese-roberta-wwm-ext-large \
#            --dump_decisions_instead
#done

#for x in fold_10_64_8927   fold_1_71_9022   fold_2_77_8992  fold_3_71_8962  fold_4_68_8986  fold_5_29_8887  fold_6_72_8982  fold_7_39_8948   fold_8_77_8943   fold_9_43_9013
#do
#    python evaluate.py \
#            --dataset ./data/coai.test_with_noise.sentences.txt \
#            --model coai_p1p2_slv1500/$x \
#            --epoch "" \
#            --gpu 5 \
#            --span_thres 0.5 \
#            --bert_path hfl/chinese-roberta-wwm-ext-large \
#            --dump_decisions_instead
#done

