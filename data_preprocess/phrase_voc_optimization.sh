mkdir -p data_out
export DATA_DIR=../../data.csig/
export OUTPUT_DIR=data_out

python phrase_vocabulary_optimization.py \
  --input_file=${DATA_DIR}/train_hq_inp_wo_context \
  --input_format=wikisplit \
  --vocabulary_size=15000 \
  --max_input_examples=1000000 \
  --output_file=${OUTPUT_DIR}/label_map.txt
