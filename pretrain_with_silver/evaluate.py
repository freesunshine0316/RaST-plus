"""Evaluate the model"""
import os
import torch
import utils
import random
import logging
import argparse
import numpy as np
import math
from data_loader import DataLoader
from SequenceTagger import BertForSequenceTagging
from metrics import f1_score, get_entities, classification_report, accuracy_score
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from nltk.translate.bleu_score import corpus_bleu
from score import Metrics

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='acl', help="Directory containing the dataset")
parser.add_argument('--model', default='acl/w_bleu_rl_transfer_token_bugfix', help="Directory containing the trained model")
parser.add_argument('--epoch', default='0', help="specific epoch for testing")
parser.add_argument('--bert_path', help="the BERT path used for training")
parser.add_argument('--gpu', default='0', help="gpu device")
parser.add_argument('--seed', type=int, default=23, help="random seed for initialization")
parser.add_argument('--span_thres', type=float, default=0.0)

def convert_tokens_to_string(tokens):
    """ Converts a sequence of tokens (string) in a single string. """
    out_string = " ".join(tokens).replace(" ##", "").strip()
    return out_string

#def convert_back_tags(pred_action, pred_start, pred_end, true_action, true_start, true_end):
#    pred_tags = []
#    true_tags = []
#    for j in range(len(pred_action)):
#        p_tags = []
#        t_tags = []
#        for i in range(len(pred_action[j])):
#            if true_action[j][i] == '-1':
#                continue
#            p_tag = pred_action[j][i]+"|"+str(pred_start[j][i])+"#"+str(pred_end[j][i])
#            p_tags.append(p_tag)
#            t_tag = true_action[j][i]+"|"+str(true_start[j][i])+"#"+str(true_end[j][i])
#            t_tags.append(t_tag)
#        pred_tags.append(p_tags)
#        true_tags.append(t_tags)
#    return pred_tags, true_tags

def convert_back_tags(source_len, pred_action, pred_start, pred_end, boundaries,
        pred_action_probs=None, pred_span_probs=None, span_thres=0.0):
    pred_tags = []
    for j in range(len(pred_action)):
        p_tags = []
        cur_src_len = int(source_len[j])
        for i in range(cur_src_len):
            if i <= boundaries[j]:
                p_tag = 'DELETE|0#0'
            elif 'args' in globals() and pred_span_probs is not None and \
                    pred_span_probs[j][i] < args.span_thres:
                p_tag = '{}|0#0'.format(pred_action[j][i])
            elif 'args' not in globals() and span_thres > 0.0 and pred_span_probs is not None and \
                    pred_span_probs[j][i] < span_thres:
                p_tag = '{}|0#0'.format(pred_action[j][i])
            else:
                p_tag = pred_action[j][i]+"|"+str(pred_start[j][i])+"#"+str(pred_end[j][i])
            p_tags.append(p_tag)
        pred_tags.append(p_tags)
    return pred_tags

def tags_to_string(source, labels, special_tokens):
    output_tokens = []
    for token, tag in zip(source, labels):
        added_phrase = tag.split("|")[1]
        start, end = added_phrase.split("#")[0], added_phrase.split("#")[1]
        if int(end) > 0 and int(end)>=int(start):
            add_phrase = source[int(start):int(end)+1]
            add_phrase = " ".join(add_phrase)
            output_tokens.append(add_phrase)
        if tag.split("|")[0]=="KEEP":
            output_tokens.append(token)

    output_tokens = " ".join(output_tokens).split()

    for tkn in special_tokens:
        while tkn in output_tokens:
            output_tokens.remove(tkn)


    if len(output_tokens)==0:
       output_tokens.append("*")
    elif len(output_tokens) > 1 and output_tokens[-1]=="*":
       output_tokens = output_tokens[:-1]
    return convert_tokens_to_string(output_tokens)

def evaluate(model, rl_model, tokenizer, data_iterator, params, epoch, mark='Eval', verbose=False, is_out_of_domain=False, span_thres=0.0,
             print_out=True):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()

    idx2tag = params.idx2tag

    true_action_tags = []
    pred_action_tags = []

    true_start_tags = []
    pred_start_tags = []

    true_end_tags = []
    pred_end_tags = []

    pred_action_probs = []
    pred_span_probs = []

    # a running average object for loss
    loss_avg = utils.RunningAverage()

    context_query_boundaries = []
    source_tokens = []
    source_len = []
    references = []

    eval_starts_tags = []
    eval_end_tags = []
    gold_starts_tags = []
    gold_end_tags = []

    for _ in range(params.eval_steps):
        # fetch the next evaluation batch
        batch_data_len, batch_data, batch_token_starts, batch_ref, batch_action, batch_start, batch_end, boundaries = next(data_iterator)
        batch_masks = batch_data != tokenizer.pad_token_id
        #print("batch data:", batch_data)
        #print("batch action:", batch_action.size())
        #print("batch reference:", len(batch_ref))

        context_query_boundaries.extend(boundaries)
        source_tokens.extend(batch_data)
        source_len.extend(batch_data_len.cpu().tolist())
        #print("len source:", len(source_tokens))

        if mark != "Infer":
            with torch.no_grad():

                xxx = model((batch_data, batch_data_len, batch_token_starts, batch_ref),
                        rl_model, token_type_ids=None, attention_mask=batch_masks,
                        labels_action=batch_action, labels_start=batch_start, labels_end=batch_end)
            loss, output = xxx[0], xxx[1:]
            loss_avg.update(loss.item())

            references.extend(batch_ref)
            #print("len references:", len(references))
        else:
            with torch.no_grad():
                output = model((batch_data, batch_data_len, batch_token_starts, batch_ref),
                               rl_model, token_type_ids=None, attention_mask=batch_masks)

        batch_action_output = output[1]
        batch_action_output = batch_action_output.detach().cpu().numpy()
        if mark != "Infer":
            batch_action = batch_action.to('cpu').numpy()

        batch_span_probs = output[2].detach().cpu().tolist() # [batch, max_len]
        pred_span_probs.extend(batch_span_probs)

        batch_start_output = output[3]
        batch_start_output = batch_start_output.detach().cpu().numpy()
        if mark != "Infer":
            batch_start = batch_start.to('cpu').numpy()

        batch_end_output = output[4]
        batch_end_output = batch_end_output.detach().cpu().numpy()
        if mark != "Infer":
            batch_end = batch_end.to('cpu').numpy()

        # print(batch_end.shape, batch_end_output.shape)
        # print(batch_start.shape, batch_start_output.shape)
        assert batch_end.shape == batch_end_output.shape
        assert batch_start.shape == batch_start_output.shape

        for batch_index in range(batch_end.shape[0]):
            eval_starts_tags.extend(batch_start_output[batch_index].tolist())
            gold_starts_tags.extend(batch_start[batch_index].tolist())
            eval_end_tags.extend(batch_end_output[batch_index].tolist())
            gold_end_tags.extend(batch_end[batch_index].tolist())

        pred_action_tags.extend([[idx2tag.get(idx) for idx in indices] for indices in batch_action_output])
        if mark != "Infer":
            true_action_tags.extend([[idx2tag.get(idx) if idx != -1 else '-1' for idx in indices] for indices in batch_action])

        pred_start_tags.extend([indices for indices in batch_start_output])
        if mark != "Infer":
            true_start_tags.extend([indices for indices in batch_start])

        pred_end_tags.extend([indices for indices in batch_end_output])
        if mark != "Infer":
            true_end_tags.extend([indices for indices in batch_end])

    pred_tags = convert_back_tags(source_len, pred_action_tags, pred_start_tags, pred_end_tags, context_query_boundaries,
            pred_action_probs=pred_action_probs, pred_span_probs=pred_span_probs, span_thres=span_thres)
    if mark != "Infer":
        true_tags = convert_back_tags(source_len, true_action_tags, true_start_tags, true_end_tags, context_query_boundaries)

    source = []
    for i in range(len(source_tokens)):
        src = tokenizer.convert_ids_to_tokens(source_tokens[i].tolist())
        assert tokenizer.pad_token not in src[:source_len[i]]
        assert src[source_len[i]:].count(tokenizer.pad_token) == len(src) - source_len[i]
        source.append(src[:source_len[i]])

    special_tokens = {tokenizer.cls_token, tokenizer.sep_token, tokenizer.unk_token, tokenizer.pad_token, '*', '|', '[SEP1]', '[SEP2]'}
    hypo = []
    for i in range(len(pred_tags)):
        #assert len(source[i])==len(pred_tags[i])
        assert len(source[i])==len(pred_tags[i])
        pred = tags_to_string(source[i], pred_tags[i], special_tokens).strip()
        hypo.append(pred.lower())
        # print("hypo: >>", pred.lower())
        # print("source: ##", source[i])
        # print("pred_tags:", pred_tags[i])

    if mark == 'Infer':
        outpath = epoch
        pred_out = open(outpath, "w")
        for i in range(len(hypo)):
            pred_out.write(hypo[i]+"\n")
        pred_out.close()
        return

    if mark == "Test":
        file_name = "/prediction_emnlp"+"_"+str(epoch)+"_.txt"
        pred_out = open(params.tagger_model_dir+file_name, "w")
        for i in range(len(hypo)):
            pred_out.write(hypo[i]+"\n")
        pred_out.close()

    if mark == "Val" and print_out:
        file_name = "/prediction_acl"+"_"+str(epoch)+"_.txt"
        pred_out = open(params.tagger_model_dir+file_name, "w")
        for i in range(len(hypo)):
            pred_out.write(hypo[i]+"\n")
        pred_out.close()

    assert len(pred_tags) == len(true_tags)

    for i in range(len(pred_tags)):
        assert len(pred_tags[i]) == len(true_tags[i])

    if is_out_of_domain:
        logging.info("***********Out-of-domain evaluation************")

    # logging loss, f1 and report
    metrics = {}
    metrics['rev_wer'] = Metrics.wer_score(references, hypo)*100.0
    bleu1, bleu2, bleu3, bleu4 = Metrics.bleu_score(references, hypo)
    em_score = Metrics.em_score(references, hypo)
    rouge1, rouge2, rougel = Metrics.rouge_score(references, hypo)
    metrics['bleu1'] = bleu1*100.0
    metrics['bleu2'] = bleu2*100.0
    metrics['bleu3'] = bleu3*100.0
    metrics['bleu4'] = bleu4*100.0
    metrics['rouge1'] = rouge1*100.0
    metrics['rouge2'] = rouge2*100.0
    metrics['rouge-L'] = rougel*100.0
    metrics['em_score'] = em_score*100.0
    f1 = f1_score(true_tags, pred_tags)
    metrics['loss'] = loss_avg()
    metrics['f1'] = f1
    accuracy = accuracy_score(true_tags, pred_tags)
    metrics['accuracy'] = accuracy
    span_start_accuracy = accuracy_score(gold_starts_tags, eval_starts_tags)
    metrics['span_start_accuracy'] = span_start_accuracy
    span_end_accuracy = accuracy_score(gold_end_tags, eval_end_tags)
    metrics['span_end_accuracy'] = span_end_accuracy
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics: ".format(mark) + metrics_str)

    if verbose:
        report = classification_report(true_tags, pred_tags)
        logging.info(report)
    return metrics

def interAct(model, data_iterator, params, mark='Interactive', verbose=False):
    """Evaluate the model on `steps` batches. Unused"""
    assert False, 'buggy function unused'
    # set model to evaluation mode
    model.eval()

    idx2tag = params.idx2tag

    true_tags = []
    pred_tags = []

    # a running average object for loss
    loss_avg = utils.RunningAverage()


    batch_data, batch_token_starts = next(data_iterator)
    batch_masks = batch_data.gt(0)

    batch_output = model((batch_data, batch_token_starts), token_type_ids=None, attention_mask=batch_masks)[0]  # shape: (batch_size, max_len, num_labels)

    batch_output = batch_output.detach().cpu().numpy()

    pred_tags.extend([[idx2tag.get(idx) for idx in indices] for indices in np.argmax(batch_output, axis=2)])

    return(get_entities(pred_tags))

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

    params.batch_size = 1

    # Set the logger
    utils.set_logger(os.path.join(tagger_model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Loading the dataset...")

    # Initialize the DataLoader
    if args.dataset in ["canard", "task", "emnlp", "acl", "coai"]:
        data_dir = 'data/' + args.dataset
        data_path = None
    else:
        data_dir = None
        data_path = args.dataset

    bert_class = args.bert_path
    print('BERT path: {}'.format(bert_class))

    data_loader = DataLoader(data_dir, bert_class, params, tag_pad_idx=-1)

    # Load the model
    tagger_model_path = os.path.join(tagger_model_dir, args.epoch)
    print(tagger_model_path)
    model = BertForSequenceTagging.from_pretrained(tagger_model_path, num_labels=len(params.tag2idx))
    model.to(params.device)

    #rl_model = GPT2LMHeadModel.from_pretrained("./dialogue_model/")
    #rl_model.to(params.device)
    #rl_model.eval()
    rl_model = None

    # Load data
    test_data = data_loader.load_data(data_type='test', data_path=data_path)
    print('Size {}'.format(test_data['size']))

    # Specify the test set size
    params.test_size = test_data['size']
    params.eval_steps = math.ceil(params.test_size / params.batch_size)
    test_data_iterator = data_loader.data_iterator(test_data, shuffle=False)

    params.tagger_model_dir = tagger_model_dir

    logging.info("- done.")

    logging.info("Starting evaluation/inferring...")
    if data_path is None:
        test_metrics = evaluate(model, rl_model, data_loader.tokenizer, test_data_iterator, params,
                epoch='Test', mark='Test', verbose=False)
    else:
        model_id = args.model.replace('/', '_')
        if args.epoch != "":
            model_id = '{}_{}'.format(model_id, args.epoch)
        pred_path = '{}_{}.pred'.format(data_path, model_id)
        #pred_path = data_path+'.pred'
        test_metrics = evaluate(model, rl_model, data_loader.tokenizer, test_data_iterator, params,
                epoch=pred_path, mark='Infer', verbose=False)
