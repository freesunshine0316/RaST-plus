export BERT_BASE_DIR=chinese_L-12_H-768_A-12
export DATA_DIR=../../data.csig
export OUTPUT_DIR=data_out

python preprocess_main_out.py \
  --input_file=${DATA_DIR}/train_hq_inp \
  --input_format=wikisplit \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --output_arbitrary_targets_for_infeasible_examples=false

for i in {1..10}
do
    python preprocess_main_out.py \
      --input_file=${DATA_DIR}/coai.train_$i\.tsv \
      --input_format=wikisplit \
      --label_map_file=${OUTPUT_DIR}/label_map.txt \
      --vocab_file=${BERT_BASE_DIR}/vocab.txt \
      --remove_unaligned=true \
      --output_arbitrary_targets_for_infeasible_examples=false
    
    mv ${OUTPUT_DIR}/sentences.txt ${DATA_DIR}/coai.train_$i\.sentences.txt
    mv ${OUTPUT_DIR}/tags.txt ${DATA_DIR}/coai.train_$i\.tags.txt
    
    python preprocess_main_out.py \
      --input_file=${DATA_DIR}/coai.dev_$i\.tsv \
      --input_format=wikisplit \
      --label_map_file=${OUTPUT_DIR}/label_map.txt \
      --vocab_file=${BERT_BASE_DIR}/vocab.txt \
      --remove_unaligned=false \
      --output_arbitrary_targets_for_infeasible_examples=false
    
    mv ${OUTPUT_DIR}/sentences.txt ${DATA_DIR}/coai.dev_$i\.sentences.txt
    mv ${OUTPUT_DIR}/tags.txt ${DATA_DIR}/coai.dev_$i\.tags.txt
done

