# RaST-plus

## Silver Data Generation

Silver data is first extracted from a large dialogue dataset, LCCC, for the purposes of pretraining and noisy finetuning.
There are two extracted silver datasets: the first one is formed by extracting the first three turns of multiturn dialogues,
and replace or drop phrases in the third turn which is also found in the first two turns. The second dataset is
one which containsthe hardest instances in the first silver dataset, which is measured by editing distance between the
original third turn and the edited third turn. Refer to the scripts in `silver_data_generation` to generate these two datasets.

## Pretraining with Silver Data

There are two steps in the pretraining phase, P1 and P2. In P1, the large silver dataset is used to further pretrain a language model.
Two tasks are used in P1, query rewriting and masked language modeling.
In P2, the hard silver dataset and the gold dataset are used to further finetune the base models with only query rewriting as 
the training task. Refer to the scripts in `pretrain_with_silver` for details.

## Main Training Step

To conduct training, first make the corresponding folder (e.g., ``coai_p1p2``) under ``experiments``.
If you use 10-fold cross validation setting, please create 10 foldes like ``fold_x``, where ``x`` belongs to \[1, 10\].
Then modify ``train.sh`` before executing ``./train.sh``. 
This training script trains only one fold at a time, so you have to manage your GPUs and launch 10 times for the training script.
