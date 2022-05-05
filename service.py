"""Evaluate the model"""
import os, json
import torch
import utils
import random
import logging
import argparse
import numpy as np
import math

from data_loader import DataLoader
from SequenceTagger import BertForSequenceTagging
from transformers import BertTokenizer

import requests
from flask import Flask, request
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='acl/w_bleu_rl_transfer_token_bugfix', help="Directory containing the trained model")
parser.add_argument('--epoch', default='0', help="specific epoch for testing")
parser.add_argument('--bert_path', help="the BERT path used for training")
parser.add_argument('--unk_list_file', default='', help="The file containing considered UNKs")
parser.add_argument('--gpu', default='0', help="gpu device")
parser.add_argument('--seed', type=int, default=23, help="random seed for initialization")
parser.add_argument('--span_thres', type=float, default=0.0)
parser.add_argument('--dump_decisions_instead', action='store_true', default=False)


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
        pred_action_probs=None, pred_span_probs=None):
    pred_tags = []
    pred_probs = []
    for j in range(len(pred_action)):
        p_tags = []
        p_probs = []
        cur_src_len = int(source_len[j])
        for i in range(cur_src_len):
            if i <= boundaries[j]:
                p_tag = 'DELETE|0#0'
                if pred_span_probs is not None:
                    p_probs.append([1.0, 1.0])
            elif 'args' in globals() and pred_span_probs is not None and \
                    pred_span_probs[j][i] < args.span_thres:
                p_tag = '{}|0#0'.format(pred_action[j][i])
                if pred_span_probs is not None:
                    p_probs.append([pred_action_probs[j][i], 1.0-pred_span_probs[j][i]])
            else:
                p_tag = pred_action[j][i]+"|"+str(pred_start[j][i])+"#"+str(pred_end[j][i])
                if pred_span_probs is not None:
                    p_probs.append([pred_action_probs[j][i], pred_span_probs[j][i]])
            p_tags.append(p_tag)
        pred_tags.append(p_tags)
        if pred_span_probs is not None:
            pred_probs.append(p_probs)
    return pred_tags, pred_probs

def tags_to_decisions(source, boundary, labels, probs):
    if 'unk_mapping_rev' in globals():
        source = [unk_mapping_rev.get(x,x) for x in source]

    decisions = []
    for i, (token, tag) in enumerate(zip(source, labels)):
        if i <= boundary or tag == 'KEEP|0#0':
            continue
        span = tag.split("|")[1]
        st, ed = int(span.split("#")[0]), int(span.split("#")[1])
        add_phrase = " ".join(source[st:ed+1])
        action = tag.split("|")[0]
        assert action in ('DELETE', 'KEEP')
        decision_str = '{} ==> {}'.format(token, add_phrase) if action == 'DELETE' \
                else '{} ==> {}{}'.format(token, add_phrase, token)
        decisions.append({i-boundary-1: decision_str})
    return decisions

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

def decode(model, rl_model, tokenizer, data_iterator, params):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()

    idx2tag = params.idx2tag

    pred_action_tags = []
    pred_start_tags = []
    pred_end_tags = []

    pred_action_probs = []
    pred_span_probs = []

    context_query_boundaries = []
    source_tokens = []
    source_len = []

    for _ in range(params.eval_steps):
        # fetch the next evaluation batch
        batch_data_len, batch_data, batch_token_starts, batch_ref, batch_action, batch_start, batch_end, boundaries = next(data_iterator)
        batch_masks = batch_data != tokenizer.pad_token_id
        #print("batch data:", batch_data)
        #print("batch action:", batch_action.size())
        #print("batch reference:", len(batch_ref))

        context_query_boundaries.extend(boundaries.detach().cpu().tolist())
        source_tokens.extend(batch_data)
        source_len.extend(batch_data_len.cpu().tolist())
        #print("len source:", len(source_tokens))

        output = model((batch_data, batch_data_len, batch_token_starts, batch_ref),
                rl_model, token_type_ids=None, attention_mask=batch_masks)

        batch_action_probs = output[0].detach().cpu().tolist() # [batch, max_len]
        pred_action_probs.extend(batch_action_probs)

        batch_action_output = output[1]
        batch_action_output = batch_action_output.detach().cpu().numpy()

        batch_span_probs = output[2].detach().cpu().tolist() # [batch, max_len]
        pred_span_probs.extend(batch_span_probs)

        batch_start_output = output[3]
        batch_start_output = batch_start_output.detach().cpu().numpy()

        batch_end_output = output[4]
        batch_end_output = batch_end_output.detach().cpu().numpy()

        pred_action_tags.extend([[idx2tag.get(idx) for idx in indices] for indices in batch_action_output])

        pred_start_tags.extend([indices for indices in batch_start_output])

        pred_end_tags.extend([indices for indices in batch_end_output])

    pred_tags, pred_probs = convert_back_tags(source_len, pred_action_tags, pred_start_tags, pred_end_tags, context_query_boundaries,
            pred_action_probs=pred_action_probs, pred_span_probs=pred_span_probs)

    source = []
    for i in range(len(source_tokens)):
        src = tokenizer.convert_ids_to_tokens(source_tokens[i].tolist())
        #assert tokenizer.pad_token not in src[:source_len[i]]
        #assert src[source_len[i]:].count(tokenizer.pad_token) == len(src) - source_len[i]
        source.append(src[:source_len[i]])

    special_tokens = set([tokenizer.cls_token, tokenizer.sep_token, tokenizer.unk_token, tokenizer.pad_token, '*', '|'])
    rewriting_results, decisions = [], []
    for i in range(len(pred_tags)):
        #print("source:", source[i])
        #print("pred_tags:", pred_tags[i])
        #assert len(source[i])==len(pred_tags[i])
        rew = tags_to_string(source[i], pred_tags[i], special_tokens).strip()
        rewriting_results.append(rew)
        dec = tags_to_decisions(source[i], context_query_boundaries[i], pred_tags[i], pred_probs[i])
        decisions.append(dec)
        #print("hypo:", rew.lower())

    return rewriting_results, decisions


class RaSTRewriter(Resource):
    def post(self):
        print("getting a post request ....")
        dialog_turns = None
        post_data = request.form.to_dict(flat=False)
        if "dialog_turns" in post_data:
            dialog_turns = post_data["dialog_turns"]
        logging.info(dialog_turns)

        if dialog_turns == None or len(dialog_turns) == 0:
            return {}

        if len(dialog_turns) == 1:
            return {'rewrite': dialog_turns[0], 'changes':[], 'origin': dialog_turns[0]}

        rewriting_results, changes = self.rewrite(dialog_turns)

        return {'rewrite': rewriting_results, 'changes': changes,
                'origin': ' '.join(RaSTRewriter.tokenize(dialog_turns[-1]))}

    def rewrite(self, dialog_turns):
        # format data
        dialog_turns_tokenized = [' '.join(RaSTRewriter.tokenize(turn)) for turn in dialog_turns]
        if len(dialog_turns_tokenized) >= 3:
            c1, c2, inp = dialog_turns_tokenized[-3:]
            inputs = '{} [SEP] {} | {} *'.format(c1, c2, inp)
        else:
            c2, inp = dialog_turns_tokenized[-2:]
            inputs = '{} | {} *'.format(c1, c2, inp)

        data = {}
        data_loader.construct_sentences_tags([inputs], data, unk_mapping=unk_mapping)
        print('Size {}'.format(data['size']))
        data_iter = data_loader.data_iterator(data)

        rewriting_results, decisions = decode(model, rl_model, data_loader.tokenizer, data_iter, params)

        return rewriting_results[0], decisions[0]

    @staticmethod
    def is_all_chinese(strs):
        for _char in strs:
            if not '\u4e00' <= _char <= '\u9fa5':
                return False
        return True

    @staticmethod
    def tokenize(sen):
        result = []
        english_token = []
        tokens = list(sen)
        for i in range(len(tokens)):
            if RaSTRewriter.is_all_chinese(tokens[i]):
                if len(english_token) > 0:
                    result.append("".join(english_token))
                    english_token = []
                result.append(tokens[i])
            else:
                english_token.append(tokens[i])
        if len(english_token)>0:
            result.append("".join(english_token))
        return result


api.add_resource(RaSTRewriter, '/rast')


if __name__ == '__main__':
    args = parser.parse_args()

    if args.unk_list_file != '':
        unk_words = json.load(open(args.unk_list_file, 'r'))
        print('UNK vocab size {}'.format(len(unk_words)))
        unk_mapping = {x:'[unused{}]'.format(i+1) for i, x in enumerate(unk_words)}
        unk_mapping_rev = {'[unused{}]'.format(i+1):x for i, x in enumerate(unk_words)}
        unk_placeholders = list(unk_mapping_rev.keys())
    else:
        unk_words = []
        print('UNK vocab size {}'.format(len(unk_words)))
        unk_mapping = {}
        unk_mapping_rev = {}
        unk_placeholders = []

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

    bert_class = args.bert_path
    print('BERT path: {}'.format(bert_class))

    data_loader = DataLoader(None, bert_class, params, tag_pad_idx=-1, lower_case=True)
    #data_loader.tokenizer.add_special_tokens({"additional_special_tokens": unk_placeholders})

    # Load the model
    tagger_model_path = os.path.join(tagger_model_dir, args.epoch)
    print(tagger_model_path)
    model = BertForSequenceTagging.from_pretrained(tagger_model_path, num_labels=len(params.tag2idx))
    model.to(params.device)

    #rl_model = GPT2LMHeadModel.from_pretrained("./dialogue_model/")
    #rl_model.to(params.device)
    #rl_model.eval()
    rl_model = None

    # Specify the test set size
    params.test_size = 1
    params.eval_steps = math.ceil(params.test_size / params.batch_size)

    params.tagger_model_dir = tagger_model_dir

    logging.info("- done.")

    logging.info('RaST rewriting service is now available')
    app.run(host='0.0.0.0', port=2206)

