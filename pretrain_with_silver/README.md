# P1 pretraining (Noisy pretrain)

P1 pretraining consists of training a language model, such as Roberta, with a large number of noisy dialogue silver data. 
There are two tasks used in P1, one of which is the target task, query rewriting, and the other is the language modeling task.
Running P1:
```bash
python pretrain.py
```
Most of the configuration settings have the same meaning as the main training script.

# P2 pretraining (Noisy finetune)

P2 pretraining follows P1 pretraining, where the hard silver training instances are used along with the gold data.
The training task is the target task, query rewriting. 
Because of use of gold training data, P2 pretraining is executed following the folds established in final clean finetuning.
Running P2:
```bash
python p2_train.py
```
Most of the configuration settings have the same meaning as the main training script.
