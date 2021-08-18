# RaST-plus

## Silver Data Generation

## Pretraining with Silver Data

## Main Training Step

To conduct training, first make the corresponding folder (e.g., ``coai_p1p2``) under ``experiments``.
If you use 10-fold cross validation setting, please create 10 foldes like ``fold_x``, where ``x`` belongs to \[1, 10\].
Then modify ``train.sh`` before executing ``./train.sh``. 
This training script trains only one fold at a time, so you have to manage your GPUs and launch 10 times for the training script.
