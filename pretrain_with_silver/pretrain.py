"""Train and evaluate the model"""
import os
import torch
import transformers

import utils
import random
import logging
import argparse
import torch.nn as nn
from tqdm import trange
from evaluate import evaluate
from data_loader import DataLoader, split_train_into_train_dev
from SequenceTagger import BertForSequenceTagging, setup_parameters_groups_with_lr_decay
from transformers.optimization import get_linear_schedule_with_warmup, AdamW
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
import math

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='bert_base', help="Directory containing the model")
parser.add_argument('--bert_path', default='None', help="Specified bert path for initialization")
parser.add_argument('--gpu', default='3', help="gpu device")
parser.add_argument('--gpt_rl', dest='gpt_rl', action='store_true', default=False, help="if use the gpt2 model for RL")
parser.add_argument('--metric_rl', dest='metric_rl', default='', help="if use a sentence-level metric (e.g., BLEU or WER) for RL")
parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, name of the directory containing weights to reload before training, e.g., 'experiments/conll/'")
parser.add_argument('--lr', default=0., type=float, help="learning rate")
parser.add_argument('--span_thres', default=0., type=float, help="span threshold")
parser.add_argument('--train_eval_frequency', default=1, type=int)
parser.add_argument('--silver_dev_size', default=3000, type=int)

parser.add_argument('--layer_lr_decay', default=1., type=float, help="learning rate")

def mask_a_batch_tokenizer(batch_ids, tokenizer):
    new_batch_id = batch_ids.clone().detach()
    for row in range(batch_ids.shape[0]):
        for col in range(batch_ids.shape[1]):
            if col == 0:
                continue
            cur_id = batch_ids[row][col]
            if cur_id == tokenizer.pad_token_id:
                break
            else:
                operation_switch = random.random()
                if operation_switch < 0.75:
                    continue
                else:
                    mask_switch = random.random()
                    if mask_switch < 0.8:
                        new_batch_id[row][col] = tokenizer.mask_token_id
                    elif 0.9 < mask_switch < 0.8:
                        new_batch_id[row][col] = random.randint(0, tokenizer.vocabsize)
                    else:
                        pass
    return new_batch_id

def mask_a_batch_tokenizer_ref(batch_ids, tokenizer):
    new_batch_id = batch_ids.clone().detach()
    star_id = tokenizer.convert_tokens_to_ids('*')
    for row in range(batch_ids.shape[0]):
        masking_sentinel = False
        for col in range(batch_ids.shape[1]):
            if col == 0:
                continue
            cur_id = batch_ids[row][col]
            if cur_id == tokenizer.pad_token_id:
                break
            else:
                if cur_id == star_id:
                    masking_sentinel = True
                else:
                    if masking_sentinel:
                        new_batch_id[row][col] = tokenizer.mask_token_id
    return new_batch_id

def train_epoch(model, rl_model, tokenizer, train_data_iterator, train_data_with_ref_iterator, optimizer, scheduler, params, lmmodel=None):
    """Train the model on `steps` batches"""
    # set model to training mode
    model.train()
    train_with_ref_THRES = 0.7
    LM_TRAIN_THRES = 0.6
    # a running average object for loss
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    one_epoch = trange(params.train_steps)
    for batch in one_epoch:
        # fetch the next training batch
        train_with_ref_switch = random.random()
        if train_with_ref_switch <= train_with_ref_THRES:
            batch_data_len, batch_data, batch_token_starts, batch_ref, batch_action, batch_start, batch_end, _ = next(train_data_iterator)
        else:
            batch_data_len, batch_data, batch_token_starts, batch_ref, batch_action, batch_start, batch_end, _ = next(train_data_with_ref_iterator)

        batch_token_masks = batch_data != tokenizer.pad_token_id # get padding mask

        batch_size, max_seq_len = list(batch_data.size())
        batch_masks_v2 = torch.arange(max_seq_len).view(1, max_seq_len).to(batch_data.device) < batch_data_len.view(batch_size, 1) # [batch, seq]
        assert torch.all(batch_token_masks == batch_masks_v2)

        # compute model output and loss
        if train_with_ref_switch > train_with_ref_THRES:    # 30% of the time train with ref always
            task_switch = random.random()
            if task_switch < LM_TRAIN_THRES:    # 18% of the time with ref training
                # print('Ref CLS')
                loss = model((batch_data, batch_data_len, batch_token_starts, batch_ref), rl_model, token_type_ids=None, attention_mask=batch_token_masks,
                         labels_action=batch_action, labels_start=batch_start, labels_end=batch_end)[0]
            else:
                labels = batch_data    # 12% of the time ref lm training
                masked_batch_data = mask_a_batch_tokenizer_ref(batch_data, tokenizer)
                # print('Masked LM ref')
                lm_output = lmmodel(masked_batch_data, attention_mask=batch_token_masks, labels=labels)
                loss = lm_output.loss
        else:
            task_switch = random.random()
            if task_switch < LM_TRAIN_THRES:    # 42% of the time normal training
                # print('NoRef, CLS')
                loss = model((batch_data, batch_data_len, batch_token_starts, batch_ref), rl_model, token_type_ids=None, attention_mask=batch_token_masks,
                         labels_action=batch_action, labels_start=batch_start, labels_end=batch_end)[0]
            else:
                labels = batch_data    # 28% of the time lm training
                masked_batch_data = mask_a_batch_tokenizer(batch_data, tokenizer)
                # print('Masked LM')
                lm_output = lmmodel(masked_batch_data, attention_mask=batch_token_masks, labels=labels)
                loss = lm_output.loss

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        # nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params.clip_grad)

        # performs updates using calculated gradients
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # update the average loss
        loss_avg.update(loss.item())
        one_epoch.set_postfix(loss='{:05.3f}'.format(loss_avg()))

def train_and_evaluate(model, rl_model, tokenizer, train_data, val_data, silver_val_data, train_data_with_ref,
        optimizer, scheduler, params, model_dir, restore_dir=None, span_thres=0.0, lmmodel=None):
    """Train the model and evaluate every epoch."""
    # reload weights from restore_dir if specified
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

        # data iterator for training
        train_data_iterator = data_loader.data_iterator(train_data, shuffle=True)
        train_data_with_ref_iterator = data_loader.data_iterator(train_data_with_ref, shuffle=True)

        # Train for one epoch on training set
        train_epoch(model, rl_model, tokenizer, train_data_iterator, train_data_with_ref_iterator, optimizer, scheduler, params, lmmodel=lmmodel)

        # data iterator for evaluation
        val_data_iterator = data_loader.data_iterator(val_data, shuffle=False)
        #test_data_iterator = data_loader.data_iterator(test_data, shuffle=False)
        #unseen_test_data_iterator = data_loader.data_iterator(unseen_test_data, shuffle=False)

        # Evaluate for one epoch on training set and validation set
        params.eval_steps = math.ceil(params.silver_val_size / params.batch_size)
        silver_val_data_iterator = data_loader.data_iterator(silver_val_data, shuffle=False)
        val_metrics = evaluate(model, rl_model, tokenizer, silver_val_data_iterator, params, epoch, mark='Val', span_thres=span_thres, print_out=False)
        silver_val_f1 = val_metrics['rev_wer']

        params.eval_steps = params.val_steps
        val_metrics = evaluate(model, rl_model, tokenizer, val_data_iterator, params, epoch, mark='Val', span_thres=span_thres, print_out=False, is_out_of_domain=True)
        val_f1 = val_metrics['rev_wer']
        improve_f1 = val_f1 - best_val_f1
        if improve_f1 > 1e-5:
            best_val_f1 = val_f1
            logging.info(f"- Found new best Rev-WER score {best_val_f1}")
        else:
            logging.info(f"- Current best is {best_val_f1}")


        # if epoch % 10 == 0:
        torch.save(model.state_dict(), model_dir+f'/epoch{epoch}-revwer{val_metrics["rev_wer"]}-silver{silver_val_f1}.pt')

        logging.info('*********************')

        # # Early stopping and logging best f1
        # if (patience_counter >= params.patience_num and epoch > params.min_epoch_num) or epoch == params.epoch_num:
        #     logging.info("Best val EM score: {:05.2f}".format(best_val_f1))
        #     break


if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    tagger_model_dir = 'pretrain_experiments/' + args.model

    # Load the parameters from json file
    json_path = os.path.join(tagger_model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPUs if available
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the random seed for reproducible experiments
    if args.seed != 0:
        seed = args.seed
    else:
        seed = random.randint(1, int(1e8))
    random.seed(seed)
    torch.manual_seed(seed)
    params.seed = seed

    # Set the logger
    utils.set_logger(os.path.join(tagger_model_dir, 'train.log'))
    logging.info("device: {}, counts {}".format(params.device, torch.cuda.device_count()))

    # Create the input data pipeline

    # Initialize the DataLoader
    data_prefix = 'data/pretrain'
    # data_prefix = 'data/coai'

    bert_class = args.bert_path
    # bert_class = 'xlm-roberta-large'
    print('BERT path: {}'.format(bert_class))

    data_loader = DataLoader(data_prefix, bert_class, params, tag_pad_idx=-1)

    logging.info("Loading the datasets...")

    # Load training data and test data
    loading_len = -1

    processed_train_path = os.path.join(tagger_model_dir, 'train.data.pt')
    processed_train_with_ref_path = os.path.join(tagger_model_dir, 'train.withref.data.pt')
    processed_val_path = os.path.join(tagger_model_dir, 'val.data.pt')

    if os.path.exists(processed_train_path):
        train_data = torch.load(processed_train_path)
        train_data_with_ref = torch.load(processed_train_with_ref_path)
        val_data = torch.load(processed_val_path)
    else:
        train_data = data_loader.load_data('train', loading_len=loading_len)
        torch.save(train_data, processed_train_path)
        train_data_with_ref = data_loader.load_data('train', with_ref=True, loading_len=loading_len)
        torch.save(train_data_with_ref, processed_train_with_ref_path)
        val_data = data_loader.load_data('dev', loading_len=loading_len)
        torch.save(val_data, processed_val_path)

    train_data, silver_val_data = split_train_into_train_dev(train_data, args.silver_dev_size)
    test_data = None  # data_loader.load_data('test')

    # Specify the training and validation dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']
    params.silver_val_size = silver_val_data['size']

    params.train_eval_frequency = args.train_eval_frequency
    logging.info(f"Datasets: Train: {params.train_size}; Val: {params.val_size}...")
    logging.info("Loading BERT model...")

    # Prepare model
    model = BertForSequenceTagging(bert_class, num_labels=len(params.tag2idx))
    lmmodel = transformers.AutoModelForMaskedLM.from_config(model.bert.config)
    lmmodel.encoder = model.bert

    model.to(params.device)
    lmmodel.to(params.device)

    rl_model = None

    if args.lr != 0:
        lr = args.lr
    else:
        lr = params.learning_rate

    # Prepare optimizer
    if params.full_finetuning:
        # if 'electra' in bert_class:
        optimizer_grouped_parameters = setup_parameters_groups_with_lr_decay(model, lr, weight_decay=params.weight_decay,
                                                                             lr_decay=args.layer_lr_decay)
        # else:
        #     param_optimizer = list(model.named_parameters())
        #     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        #     optimizer_grouped_parameters = [
        #         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        #          'weight_decay': params.weight_decay},
        #         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        #          'weight_decay': 0.0}
        #     ]
    else: # only finetune the head classifier
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer], 'lr':lr}]

    optimizer = AdamW(optimizer_grouped_parameters, correct_bias=False)
    # optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)

    train_steps_per_epoch = math.ceil(params.train_size / params.batch_size) // params.train_eval_frequency
    logging.info(f"{params.epoch_num} epochs. Each epoch as {train_steps_per_epoch} instances...")

    total_training_steps = params.epoch_num * train_steps_per_epoch
    warm_up_steps = int(train_steps_per_epoch * 5)  # first 5 epochs are warmups
    logging.info(f"{warm_up_steps} instances for warmup...")

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps, num_training_steps=total_training_steps)
    # scheduler = None

    params.tagger_model_dir = tagger_model_dir
    # Train and evaluate the model
    logging.info("Starting training for {} epoch(s)".format(params.epoch_num))
    train_and_evaluate(model, rl_model, data_loader.tokenizer, train_data, val_data, silver_val_data, train_data_with_ref,
            optimizer, scheduler, params, tagger_model_dir, args.restore_dir, args.span_thres, lmmodel=lmmodel)

