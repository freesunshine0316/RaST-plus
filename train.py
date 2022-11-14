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
from data_loader import DataLoader
from SequenceTagger import BertForSequenceTagging
from torch.optim import Adam
from transformers.optimization import get_linear_schedule_with_warmup, AdamW
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
import math

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='acl', help="Directory containing the dataset")
parser.add_argument('--fold', type=str, help="The fold of dataset")
parser.add_argument('--model', default='acl/w_bleu_rl_transfer_token_bugfix', help="Directory containing the model")
parser.add_argument('--bert_path', default='None', help="Specified bert path for initialization")
parser.add_argument('--lower_case', action='store_true', default=False)
parser.add_argument('--gpu', default='3', help="gpu device")
parser.add_argument('--gpt_rl', dest='gpt_rl', action='store_true', default=False, help="if use the gpt2 model for RL")
parser.add_argument('--metric_rl',
                    dest='metric_rl',
                    default='',
                    help="if use a sentence-level metric (e.g., BLEU or WER) for RL")
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
parser.add_argument(
    '--restore_point',
    default=None,
    help="Optional, name of the directory containing weights to reload before training, e.g., 'experiments/conll/'")


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
        batch_data_len, batch_data, batch_token_starts, batch_ref, batch_action, batch_start, batch_end, _ = next(
            data_iterator)
        batch_masks = batch_data != tokenizer.pad_token_id  # get padding mask

        batch_size, max_seq_len = list(batch_data.size())
        batch_masks_v2 = torch.arange(max_seq_len).view(1, max_seq_len).to(batch_data.device) < batch_data_len.view(
            batch_size, 1)  # [batch, seq]
        assert torch.all(batch_masks == batch_masks_v2)

        # compute model output and loss
        loss = model((batch_data, batch_data_len, batch_token_starts, batch_ref),
                     rl_model,
                     token_type_ids=None,
                     attention_mask=batch_masks,
                     labels_action=batch_action,
                     labels_start=batch_start,
                     labels_end=batch_end)[0]

        # update the average loss
        loss_avg.update(loss.item())
        one_epoch.set_postfix(loss='{:05.3f}'.format(loss_avg()))

        if params.grad_accum_steps > 1:
            loss = loss / params.grad_accum_steps

        # compute gradients of all variables wrt loss
        loss.backward()

        if batch % params.grad_accum_steps == 0 or batch == params.train_steps - 1:
            # gradient clipping
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params.clip_grad)

            # performs updates using calculated gradients
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            model.zero_grad()


def train_and_evaluate(model, rl_model, tokenizer, train_data, val_data, test_data, unseen_test_data, optimizer,
                       scheduler, params, model_dir):
    """Train the model and evaluate every epoch."""
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, params.epoch_num + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, params.epoch_num))

        # Compute number of batches in one epoch
        params.train_steps = math.ceil(params.train_size / params.batch_size)
        params.val_steps = math.ceil(params.val_size / params.batch_size)
        params.test_steps = math.ceil(params.test_size / params.batch_size)
        #params.unseen_test_steps = math.ceil(params.unseen_test_size / params.batch_size)

        # data iterator for training
        train_data_iterator = data_loader.data_iterator(train_data, shuffle=True)

        # Train for one epoch on training set
        train_epoch(model, rl_model, tokenizer, train_data_iterator, optimizer, scheduler, params)

        # data iterator for evaluation
        val_data_iterator = data_loader.data_iterator(val_data, shuffle=False)
        test_data_iterator = data_loader.data_iterator(test_data, shuffle=False)
        #unseen_test_data_iterator = data_loader.data_iterator(unseen_test_data, shuffle=False)

        # Evaluate for one epoch on training set and validation set
        params.eval_steps = params.val_steps
        val_metrics = evaluate(model, rl_model, tokenizer, val_data_iterator, params, epoch, mark='Val-Pos')

        val_f1 = val_metrics['rev_wer']
        improve_f1 = val_f1 - best_val_f1
        if improve_f1 > 1e-5:
            logging.info("- Found new best Rev-WER score")
            best_val_f1 = val_f1
            if os.path.exists(model_dir + "/" + str(epoch)) is False:
                os.mkdir(model_dir + "/" + str(epoch))
            model.save_pretrained(model_dir + "/" + str(epoch))
        # Early stop
        #if improve_f1 < params.patience:
        #    patience_counter += 1
        #else:
        #    patience_counter = 0
        #if patience_counter > 10:
        #    break

        # Test data evaluation
        params.eval_steps = params.test_steps
        evaluate(model, rl_model, tokenizer, test_data_iterator, params, epoch, mark='Val-Neg')

        #params.eval_steps = params.unseen_test_steps
        #test_metrics = evaluate(model, rl_model, tokenizer, unseen_test_data_iterator, params, epoch, mark='Test', is_out_of_domain=True)

        logging.info('*********************')

        # Early stopping and logging best f1
        if (patience_counter >= params.patience_num and epoch > params.min_epoch_num) or epoch == params.epoch_num:
            logging.info("Best val EM score: {:05.2f}".format(best_val_f1))
            break


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
    data_prefix = 'data_preprocess/data/' + args.dataset

    bert_class = args.bert_path
    print('BERT path: {}'.format(bert_class))

    data_loader = DataLoader(data_prefix, bert_class, params, tag_pad_idx=-1, lower_case=args.lower_case)
    if data_loader.tokenizer.pad_token_id != 0:
        print('!!!WARNING pad_id != 0 may cause severe issue')

    logging.info("Loading the datasets...")

    # Load training data and test data
    train_data = data_loader.load_data(f'train_{args.fold}')
    val_data = data_loader.load_data('dev_pos')
    test_data = data_loader.load_data('dev_neg')
    unseen_test_data = None  # data_loader.load_data('unseen_test')

    # Specify the training and validation dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']
    params.test_size = test_data['size']
    #params.unseen_test_size = unseen_test_data['size']

    logging.info("Loading BERT model...")

    # Prepare model
    model = BertForSequenceTagging.from_pretrained(bert_class, num_labels=len(params.tag2idx))
    model.set_tokenizer(bert_class, args.lower_case)
    model.to(params.device)
    if args.restore_point != None:
        logging.info("Found restore checkpoint {} ...".format(args.restore_point))
        model.bert.load_state_dict(torch.load(args.restore_point, map_location=params.device))

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

    # Prepare optimizer
    if params.full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': params.weight_decay
        }, {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }]
    else:  # only finetune the head classifier
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]

    #optimizer = Adam(optimizer_grouped_parameters, lr=params.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=params.learning_rate, correct_bias=False)
    train_steps_per_epoch = math.ceil(params.train_size / params.batch_size / params.grad_accum_steps)
    train_steps_total = params.epoch_num * train_steps_per_epoch
    scheduler = None
    if params.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=3 * train_steps_per_epoch,
                                                    num_training_steps=3 * train_steps_total)
    print('Scheduler: {}'.format(scheduler))

    params.tagger_model_dir = tagger_model_dir
    # Train and evaluate the model
    logging.info("Starting training for {} epoch(s)".format(params.epoch_num))
    train_and_evaluate(model, rl_model, data_loader.tokenizer, train_data, val_data, test_data, unseen_test_data,
                       optimizer, scheduler, params, tagger_model_dir)
