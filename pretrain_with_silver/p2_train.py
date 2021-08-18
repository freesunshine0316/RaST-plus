"""Train and evaluate the model"""
import os
import torch
import utils
import random
import logging
import argparse
import torch.nn as nn
from tqdm import trange
from evaluate import evaluate
from data_loader import DataLoader, merge_trains_into_train
from SequenceTagger import BertForSequenceTagging, setup_parameters_groups_with_lr_decay
from transformers.optimization import get_linear_schedule_with_warmup, AdamW
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
import math

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cross-_validation_data/data/coai', help="Directory containing the dataset")
parser.add_argument('--model', default='acl/w_bleu_rl_transfer_token_bugfix', help="Directory containing the model")
parser.add_argument('--bert_path', default='None', help="Specified bert path for initialization")
parser.add_argument('--gpu', default='3', help="gpu device")
parser.add_argument('--gpt_rl', dest='gpt_rl', action='store_true', default=False, help="if use the gpt2 model for RL")
parser.add_argument('--metric_rl', dest='metric_rl', default='', help="if use a sentence-level metric (e.g., BLEU or WER) for RL")
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, name of the directory containing weights to reload before training, e.g., 'experiments/conll/'")
parser.add_argument('--lr', default=0., type=float, help="learning rate")
parser.add_argument('--span_thres', default=0., type=float, help="span threshold")
parser.add_argument('--layer_lr_decay', default=1., type=float, help="learning rate")
parser.add_argument('--pretrained_model', default='', type=str, help="learning rate")
parser.add_argument('--fold', default='0', type=str, help="learning rate")

parser.add_argument('--train_eval_frequency', default=1, type=int)
parser.add_argument('--hardtrain_num_epochs', default=0, type=int)
parser.add_argument('--noisy_clean_mixup_portion', default=0., type=float)

def train_epoch(model, rl_model, tokenizer, data_iterator, optimizer, scheduler, params):
    """Train the model on `steps` batches"""
    # set model to training mode
    model.train()

    # a running average object for loss
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    one_epoch = trange(params.train_steps)
    for batch in one_epoch:
        # fetch the next training batch
        batch_data_len, batch_data, batch_token_starts, batch_ref, batch_action, batch_start, batch_end, _ = next(data_iterator)
        batch_masks = batch_data != tokenizer.pad_token_id # get padding mask

        batch_size, max_seq_len = list(batch_data.size())
        batch_masks_v2 = torch.arange(max_seq_len).view(1, max_seq_len).to(batch_data.device) < batch_data_len.view(batch_size, 1) # [batch, seq]
        assert torch.all(batch_masks == batch_masks_v2)

        # compute model output and loss
        loss = model((batch_data, batch_data_len, batch_token_starts, batch_ref), rl_model, token_type_ids=None, attention_mask=batch_masks,
                     labels_action=batch_action, labels_start=batch_start, labels_end=batch_end)[0]

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params.clip_grad)

        # performs updates using calculated gradients
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # update the average loss
        loss_avg.update(loss.item())
        one_epoch.set_postfix(loss='{:05.3f}'.format(loss_avg()))


def train_and_evaluate(model, rl_model, tokenizer, train_data, val_data, test_data, unseen_test_data,
        optimizer, scheduler, params, model_dir, restore_dir=None, span_thres=0.0, fold='1'):
    """Train the model and evaluate every epoch."""
    # reload weights from restore_dir if specified
    logging.info(f'fold {fold} is being used as train.')
    if restore_dir is not None:
        model = BertForSequenceTagging.from_pretrained(restore_dir)
        model.to(params.device)

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, params.epoch_num + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, params.epoch_num))

        # Compute number of batches in one epoch
        params.train_steps = math.ceil(params.train_size / params.batch_size) // params.train_eval_frequency
        params.val_steps = math.ceil(params.val_size / params.batch_size)
        #params.test_steps = math.ceil(params.test_size / params.batch_size)
        #params.unseen_test_steps = math.ceil(params.unseen_test_size / params.batch_size)

        # ###############
        # # data iterator for evaluation
        # val_data_iterator = data_loader.data_iterator(val_data, shuffle=False)
        # #test_data_iterator = data_loader.data_iterator(test_data, shuffle=False)
        # #unseen_test_data_iterator = data_loader.data_iterator(unseen_test_data, shuffle=False)
        #
        # # Evaluate for one epoch on training set and validation set
        # params.eval_steps = params.val_steps
        # val_metrics = evaluate(model, rl_model, tokenizer, val_data_iterator, params, epoch, mark='Val', span_thres=span_thres)
        # ##########

        # data iterator for training
        train_data_iterator = data_loader.data_iterator(train_data, shuffle=True)

        # Train for one epoch on training set
        train_epoch(model, rl_model, tokenizer, train_data_iterator, optimizer, scheduler, params)

        # data iterator for evaluation
        val_data_iterator = data_loader.data_iterator(val_data, shuffle=False)
        #test_data_iterator = data_loader.data_iterator(test_data, shuffle=False)
        #unseen_test_data_iterator = data_loader.data_iterator(unseen_test_data, shuffle=False)

        # Evaluate for one epoch on training set and validation set
        params.eval_steps = params.val_steps
        val_metrics = evaluate(model, rl_model, tokenizer, val_data_iterator, params, epoch, mark='Val', span_thres=span_thres)

        val_f1 = val_metrics['rev_wer']
        improve_f1 = val_f1 - best_val_f1
        if improve_f1 > 1e-5:
            best_val_f1 = val_f1
            logging.info(f"- Found new best Rev-WER score {best_val_f1}")
        else:
            logging.info(f"- Current best is {best_val_f1}")

        torch.save(model.bert.state_dict(), f'pretrain_experiments_p1p2/roberta_large/fold{fold}-e{epoch}-wr{best_val_f1}.pt')

        # if os.path.exists(model_dir+"/"+str(epoch)) == False:
        #     os.mkdir(model_dir+"/"+str(epoch))
        # model.save_pretrained(model_dir+"/"+str(epoch))

        # Early stop
        #if improve_f1 < params.patience:
        #    patience_counter += 1
        #else:
        #    patience_counter = 0
        #if patience_counter > 10:
        #    break

        ## Test data evaluation
        #params.eval_steps = params.test_steps
        #test_metrics = evaluate(model, rl_model, tokenizer, test_data_iterator, params, epoch, mark='Test')

        #params.eval_steps = params.unseen_test_steps
        #test_metrics = evaluate(model, rl_model, tokenizer, unseen_test_data_iterator, params, epoch, mark='Test', is_out_of_domain=True)

        logging.info('*********************')

        # Early stopping and logging best f1
        if (patience_counter >= params.patience_num and epoch > params.min_epoch_num) or epoch == params.epoch_num:
            logging.info("Best val EM score: {:05.2f}".format(best_val_f1))
            break
    return best_val_f1

if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    tagger_model_dir = 'experiments/' + args.model

    # Load the parameters from json file
    json_path = os.path.join(tagger_model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPUs if available
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    params.seed = args.seed

    # Set the logger
    utils.set_logger(os.path.join(tagger_model_dir, 'train.log'))
    logging.info("device: {}, counts {}".format(params.device, torch.cuda.device_count()))

    # Create the input data pipeline

    # Initialize the DataLoader
    data_prefix = 'data/' + args.dataset

    bert_class = args.bert_path
    # bert_class = 'xlm-roberta-large'
    print('BERT path: {}'.format(bert_class))

    data_loader = DataLoader(data_prefix, bert_class, params, tag_pad_idx=-1)

    logging.info("Loading the datasets...")

    using_ernie = False
    if 'ernie' in bert_class:
        SPECIAL_TOKENS = {'additional_special_tokens': ['[SEP1]', '[SEP2]']}
        using_ernie = True

        data_loader.tokenizer.add_special_tokens(SPECIAL_TOKENS)

    # Load training data and test data
    train_data = data_loader.load_data('train', fold=args.fold, using_ernie=using_ernie)
    val_data = data_loader.load_data('dev', fold=args.fold, using_ernie=using_ernie)
    test_data = None  # data_loader.load_data('test')
    unseen_test_data = None  # data_loader.load_data('unseen_test')

    noisy_finetuning = False
    noisy_mixup = False
    if args.hardtrain_num_epochs > 0:
        noisy_finetuning = True
    if args.noisy_clean_mixup_portion > 0:
        noisy_mixup = True

    # Specify the training and validation dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']
    params.train_eval_frequency = args.train_eval_frequency
    #params.test_size = test_data['size']
    #params.unseen_test_size = unseen_test_data['size']

    logging.info("Loading BERT model...")

    # Prepare model
    model = BertForSequenceTagging(bert_class, num_labels=len(params.tag2idx))
    if args.pretrained_model:
        logging.info(f'Loading pretrained model {args.pretrained_model}')
        with open(args.pretrained_model, 'rb') as saved_fh:
            state_dict = torch.load(saved_fh, map_location='cpu')
        model.load_state_dict(state_dict)
        model.reset_classifiers()
    if 'ernie' in bert_class:
        model.bert.resize_token_embeddings(len(data_loader.tokenizer))
    model.to(params.device)

    assert not (args.gpt_rl and args.metric_rl != ''), '!!!RL with GPT or Metric, cannot take both'
    if args.gpt_rl:
        print("Using GPT2 PPL as the rewards for RL training!")
        rl_model = GPT2LMHeadModel.from_pretrained("./dialogue_model/")
        rl_model.to(params.device)
        rl_model.eval()
    elif args.metric_rl != '':
        rl_model = args.metric_rl
    else:
        rl_model = None
    print('RL model: {}'.format(rl_model))

    if args.lr != 0:
        lr = args.lr
    else:
        lr = params.learning_rate

    # Prepare optimizer
    if params.full_finetuning:
        optimizer_grouped_parameters = setup_parameters_groups_with_lr_decay(model, lr, weight_decay=params.weight_decay,
                                                                             lr_decay=args.layer_lr_decay)
        # param_optimizer = list(model.named_parameters())
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        #      'weight_decay': params.weight_decay},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        #      'weight_decay': 0.0}
        # ]
    else: # only finetune the head classifier
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer], 'lr':lr}]
    params.tagger_model_dir = tagger_model_dir

    num_clean_train_epochs = params.epoch_num

    # noisy finetuning
    if noisy_finetuning is True:
        hard_silver_train_data = data_loader.load_data('train', hard=True)
        noisy_finetuning_train_data = merge_trains_into_train(train_data, hard_silver_train_data)
        params.train_size = noisy_finetuning_train_data['size']
        params.val_size = val_data['size']
        params.epoch_num = args.hardtrain_num_epochs
        noisy_optimizer = AdamW(optimizer_grouped_parameters, correct_bias=False)
        noisy_train_steps_per_epoch = math.ceil(noisy_finetuning_train_data['size'] / params.batch_size)
        noisy_total_training_steps = args.hardtrain_num_epochs * noisy_train_steps_per_epoch * 3
        noisy_warm_up_steps = noisy_train_steps_per_epoch  # 1 epoch of warm up, decay slow 3 times
        noisy_scheduler = get_linear_schedule_with_warmup(noisy_optimizer, num_warmup_steps=noisy_warm_up_steps, num_training_steps=noisy_total_training_steps)
        logging.info("Starting noisy finetuning for {} epoch(s)".format(args.hardtrain_num_epochs))
        train_and_evaluate(model, rl_model, data_loader.tokenizer, noisy_finetuning_train_data, val_data, test_data, unseen_test_data,
                           noisy_optimizer, noisy_scheduler, params, tagger_model_dir, args.restore_dir, args.span_thres, fold=args.fold)

        # reset classifiers
        # model.reset_classifiers()
    # exit(-1)
    #
    # if noisy_mixup is True:
    #     if not noisy_finetuning:
    #         hard_silver_train_data = data_loader.load_data('train', hard=True)
    #     train_data = merge_trains_into_train(train_data, hard_silver_train_data, portion=noisy_mixup)
    #
    # # clean finetuning
    # params.epoch_num = num_clean_train_epochs
    # optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)
    # # optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    # params.train_size = train_data['size']
    # params.val_size = val_data['size']
    # params.train_eval_frequency = args.train_eval_frequency
    #
    # train_steps_per_epoch = math.ceil(params.train_size / params.batch_size)
    # total_training_steps = params.epoch_num * train_steps_per_epoch * 3
    # warm_up_steps = train_steps_per_epoch * 3
    #
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps, num_training_steps=total_training_steps)
    # # scheduler = None
    #
    # # Train and evaluate the model
    # logging.info("Starting training for {} epoch(s)".format(params.epoch_num))
    # train_and_evaluate(model, rl_model, data_loader.tokenizer, train_data, val_data, test_data, unseen_test_data,
    #         optimizer, scheduler, params, tagger_model_dir, args.restore_dir, args.span_thres)

