from transformers.models.bert.modeling_bert import *
from transformers import BertTokenizer
import transformers
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch
import torch.nn as nn
import os, sys, json, codecs
import re
#from multi_headed_attn import MultiHeadedAttention
from multi_headed_additive_attn import MultiHeadedAttention
from torch.distributions import Categorical
import logging

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
cc = SmoothingFunction()

from jiwer import wer

#import score_function

import math

def rev_wer(ref, hypo):
    score = 1.0 - min(wer(ref, hypo), 1.0)
    #print('{}\n{}\n-----\n'.format(ref, hypo, score))
    return score

# span classifier based on self-attention
class SpanClassifier(nn.Module):
    def __init__(self, hidden_dim, max_relative_position):
        super(SpanClassifier, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.span_st_attn = MultiHeadedAttention(1, hidden_dim, max_relative_positions=max_relative_position)
        self.span_ed_attn = MultiHeadedAttention(1, hidden_dim, max_relative_positions=max_relative_position)
        if max_relative_position > 0.0:
            print("Setting max_relative_position to {}".format(max_relative_position))

    def forward(self, repre, mask):
        #repre = self.layer_norm(repre)

        #tmp1 = mask.unsqueeze(1) # [batch, 1, seq]
        #tmp2 = tmp1.transpose(1, 2) # [batch, seq, 1]
        #square_mask = tmp2.matmul(tmp1).byte() # [batch, seq, seq]
        #square_mask = ~square_mask
        square_mask = mask
        span_st_dist = self.span_st_attn(repre, repre, repre,
                mask=square_mask, type="self") # [batch, seq, seq]
        span_ed_dist = self.span_ed_attn(repre, repre, repre,
                mask=square_mask, type="self") # [batch, seq, seq]
        return span_st_dist, span_ed_dist

# dist: [batch, seq, seq]
# refs: [batch, seq, seq]
# masks: [batch, seq]
def token_classification_loss_v2(dist, refs, masks):
    loss = torch.sum(dist * refs.float(), dim=-1) # [batch, seq]
    num_tokens = torch.sum(masks).item()
    #assert num_tokens > 1
    return -1.0 * torch.sum(loss * masks) / num_tokens if num_tokens > 0 else torch.sum(loss * 0.0)

# start_dist: [batch, seq, seq]
# end_dist: [batch, seq, seq]
# start_positions: [batch, seq, seq]
# end_positions: [batch, seq, seq]
# seq_masks: [batch, seq]
def span_loss(start_dist, end_dist, start_positions, end_positions, seq_masks):
    span_st_loss = token_classification_loss_v2(start_dist, start_positions, seq_masks)
    span_ed_loss = token_classification_loss_v2(end_dist, end_positions, seq_masks)
    return span_st_loss + span_ed_loss

def clip_and_normalize(word_probs, epsilon):
    word_probs = torch.clamp(word_probs, epsilon, 1.0 - epsilon)
    return word_probs / word_probs.sum(dim=-1, keepdim=True)

def merge_bpe_and_make_string(tokens):
    """ Converts a sequence of tokens (string) in a single string. """
    out_string = " ".join(tokens).replace(" ##", "").strip()
    return out_string

def decode_into_string(source, label_action, label_start, label_end, label_mask):
    assert len(source) == len(label_action)

    labels = []
    action_map = {0:"KEEP", 1:"DELETE"}

    for idx in range(0, len(label_action)):
        if label_mask[idx]:
            if label_end[idx] ==0 or label_start[idx] > label_end[idx]:
                st = 0
                ed = 0
            else:
                st = label_start[idx]
                ed = label_end[idx]
            labels.append(action_map[label_action[idx]]+"|"+str(st)+"#"+str(ed))
        else:
            labels.append('DELETE')
    output_tokens = []
    for token, tag in zip(source, labels):
        if len(tag.split("|"))>1:
            added_phrase = tag.split("|")[1]
            start, end = added_phrase.split("#")[0], added_phrase.split("#")[1]
            if int(end) != 0 and int(end)>=int(start):
                add_phrase = source[int(start):int(end)+1]
                add_phrase = " ".join(add_phrase)
                output_tokens.append(add_phrase)
            if tag.split("|")[0]=="KEEP":
                output_tokens.append(token)

    output_tokens = " ".join(output_tokens).split()

    special_tokens = set([tokenizer.cls_token, tokenizer.sep_token, tokenizer.unk_token, tokenizer.pad_token, '*', ''])
    for tkn in special_tokens:
        while tkn in output_tokens:
            output_tokens.remove(tkn)

    if len(output_tokens)==0:
        output_tokens.append("*")
    elif len(output_tokens) > 1 and output_tokens[-1]=="*":
        output_tokens = output_tokens[:-1]
    return merge_bpe_and_string(output_tokens)


class BertForSequenceTagging(nn.Module):
    def __init__(self, encoder_name, num_labels):
        super().__init__()
        self.num_labels = num_labels

        self.bert = transformers.AutoModel.from_pretrained(encoder_name)
        self.bert_config = self.bert.config
        self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert_config.hidden_size, self.bert_config.num_labels)

        self.span_classifier = SpanClassifier(self.bert_config.hidden_size, 0.0)
        self._rl_ratio = 0.5
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(encoder_name, do_lower_case=False)
        self._tokenizer = BertTokenizer.from_pretrained("./dialogue_model/")

        # self.bert.init_weights()
    def reset_classifiers(self):
        logging.info('RESETTING THE CLASSIFICATION LAYERS.')
        for name, param in self.named_parameters():
            if 'bert' not in name:
                torch.nn.init.normal_(param.data, mean=0, std=0.02)

    def forward(self, input_data, rl_model, token_type_ids=None, attention_mask=None, labels_action=None,
            labels_start=None, labels_end=None, position_ids=None, inputs_embeds=None, head_mask=None, boundaries=None):
        input_ids, input_ids_len, input_token_starts, input_ref = input_data
        batch_size, max_seq_len = list(input_ids.size())
        #print("input_ids", input_ids.shape)
        #print("input_token_starts", input_token_starts.shape)
        #print("attention_mask", attention_mask.shape)
        #print("labels", labels.shape)
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        #sequence_output = self.dropout(sequence_output)
        #print("sequence_output", sequence_output.shape)

        if rl_model == None:
            self._rl_ratio = 0.0
        logits = self.classifier(sequence_output) #[bs, seq, 2]

        if labels_action is not None:
            span_clf_mask = (labels_action != -1).float()
        else:
            span_clf_mask = attention_mask.float()

        log_start_dist, log_end_dist = self.span_classifier(sequence_output, span_clf_mask)
        if boundaries is not None:
            # bound_mask = torch.arange(max_seq_len).view(1, max_seq_len) >= boundaries.view(batch_size, 1)
            # bound_mask = bound_mask.to(input_ids.device).unsqueeze(dim=2) # [batch, seq, 1]
            start_dist_clip = log_start_dist  #.masked_fill(bound_mask, 0.0) # [batch, seq, seq]
            end_dist_clip = log_end_dist  #.masked_fill(bound_mask, 0.0) # [batch ,seq, seq]
            dist_clip = start_dist_clip.unsqueeze(dim=3) * end_dist_clip.unsqueeze(dim=2) # [batch, seq, seq-st, seq-ed]
            dist_clip = dist_clip * torch.triu(torch.ones(max_seq_len, max_seq_len)).to(dist_clip.device).view(1, 1, max_seq_len, max_seq_len)

            span_probs, span_outputs = dist_clip.view(batch_size, max_seq_len, max_seq_len * max_seq_len).max(dim=2) # [batch, seq]
            start_outputs = span_outputs // max_seq_len # [batch, seq]
            end_outputs = span_outputs % max_seq_len # [batch, seq]
        else:
            log_start_probs, start_outputs = log_start_dist.max(dim=-1) # [batch, seq]
            log_end_probs, end_outputs = log_end_dist.max(dim=-1) # [batch, seq]
            span_probs = log_start_probs.exp() * log_end_probs.exp() # [batch, seq]

        act_probs, act_outputs = F.softmax(logits, dim=2).max(dim=2)
        outputs = (act_probs, act_outputs, span_probs, start_outputs, end_outputs)
        if labels_action is not None:
            labels_start = torch.nn.functional.one_hot(labels_start, num_classes=list(labels_start.size())[-1]) #[bs, seq, seq]
            labels_end = torch.nn.functional.one_hot(labels_end, num_classes=list(labels_end.size())[-1]) #[bs, seq, seq]
            loss_span = span_loss(log_start_dist, log_end_dist, labels_start, labels_end, labels_action.gt(-1).float())
            loss_mask = labels_action.gt(-1)
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if loss_mask is not None:
                active_loss = loss_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels_action.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss + loss_span

            if self._rl_ratio > 0.0:
                samples_action = Categorical(logits=logits).sample() # [bs, seq]
                samples_start = Categorical(probs=log_start_dist).sample() # [bs, seq]
                samples_end = Categorical(probs=log_end_dist).sample() # [bs, seq]
                samples_action_prob = torch.gather(logits, 2, samples_action.unsqueeze(dim=2)) #[bs, seq_len, 1]
                samples_start_prob = torch.gather(log_start_dist, 2, samples_start.unsqueeze(dim=2)) #[bs, seq_len, 1]
                samples_end_prob = torch.gather(log_end_dist, 2, samples_end.unsqueeze(dim=2)) #[bs, seq_len, 1]

                samples_action_prob = samples_action_prob.unsqueeze(dim=2)
                samples_start_prob = samples_start_prob.unsqueeze(dim=2)
                samples_end_prob = samples_end_prob.unsqueeze(dim=2)

                greedy_action = logits.argmax(dim=-1) # [bs, seq]
                greedy_start = start_outputs # [bs, seq]
                greedy_end = end_outputs # [bs, seq]

                rewards = []
                samples_mask = labels_action.gt(-1).float()

                #samples_mask_v2 = torch.arange(max_seq_len).view(1, max_seq_len).to(input_ids.device) < input_ids_len.view(batch_size, 1) # [bs, seq]
                #samples_mask_v2 = samples_mask_v2.float()
                #assert torch.all(samples_mask == samples_mask_v2)

                def Gpt_score(sentence):
                    tokenize_input = self._tokenizer.tokenize(sentence)
                    if len(tokenize_input)>300:
                        tokenize_input = tokenize_input[:300]
                    tensor_input = torch.tensor([self._tokenizer.convert_tokens_to_ids(tokenize_input)])
                    tensor_input = tensor_input.cuda()
                    outputs = gpt_model(input_ids=tensor_input, labels=tensor_input)
                    loss = outputs[0]
                    if math.exp(loss) >0.0:
                        ppl = loss
                    else:
                        return 0.0
                    b = 5.92+3*1.84
                    a = 5.92-3*1.84
                    #b = 6.24+3*1.99
                    #a = 6.24-3*1.99
                    if ppl > b:
                        ppl_norm = 1.0
                    elif ppl < a:
                        ppl_norm = 0.0
                    else:
                        ppl_norm = (b-ppl)/(b-a)
                    return ppl_norm

                for i in range(len(samples_start)):
                    weight = (0.25, 0.25, 0.25, 0.25)
                    input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i].tolist())
                    sample_str = decode_into_string(input_tokens, samples_action[i].tolist(), samples_start[i].tolist(), samples_end[i].tolist(), samples_mask[i].tolist())
                    greedy_str = decode_into_string(input_tokens, greedy_action[i].tolist(), greedy_start[i].tolist(), greedy_end[i].tolist(), samples_mask[i].tolist())
                    #print('{}\n{}\n-----\n'.format(sample_str, greedy_str))
                    if type(rl_model) is str:
                        if rl_model == 'bleu':
                            sample_score = sentence_bleu([input_ref[i].split()], sample_str.split(), weights=weight, smoothing_function=cc.method3)
                            greedy_score = sentence_bleu([input_ref[i].split()], greedy_str.split(), weights=weight, smoothing_function=cc.method3)
                        elif rl_model == 'wer':
                            sample_score = rev_wer(input_ref[i], sample_str)
                            greedy_score = rev_wer(input_ref[i], greedy_str)
                        else:
                            assert False, 'unsupported metric for RL: {}'.format(rl_model)
                    elif rl_model is not None:
                        sample_score = Gpt_score(sample_str)
                        greedy_score = Gpt_score(greedy_str)
                    else:
                        assert False
                    rewards.append(sample_score-greedy_score)

                rewards = torch.tensor(rewards).cuda()

                loss_action_rl = -1.0 * clip_and_normalize(samples_action_prob, 1e-6).log()*rewards.unsqueeze(dim=1)*samples_mask
                loss_action_rl = loss_action_rl.sum()/samples_mask.sum()
                loss_st_rl = -1.0 * clip_and_normalize(samples_start_prob, 1e-6).log()*rewards.unsqueeze(dim=1)*samples_mask
                loss_st_rl = loss_st_rl.sum()/samples_mask.sum()
                loss_ed_rl = -1.0 * clip_and_normalize(samples_end_prob, 1e-6).log()*rewards.unsqueeze(dim=1)*samples_mask
                loss_ed_rl = loss_ed_rl.sum()/samples_mask.sum()

                loss_rl = loss_action_rl + loss_st_rl + loss_ed_rl
                loss = (1.0 - self._rl_ratio) * loss + self._rl_ratio * loss_rl

            outputs = (loss,) + outputs

        return outputs  # (loss), scores

def setup_parameters_groups_with_lr_decay(model, max_lr, lr_decay=0.8, weight_decay=0.0, lmmodel=None):
    params_optimization = list(model.named_parameters())
    no_weight_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    num_encoder_layers = model.bert.config.num_hidden_layers

    param_lr_weight_decay_dict = {}

    for layer in range(num_encoder_layers - 1, -1, -1):
        key1 = f'layer.{layer}-nwd'
        key2 = f'layer.{layer}-wd'
        dic1 = {'params':[], 'weight_decay':0., 'lr':max_lr*lr_decay**(num_encoder_layers - 1 - layer)}
        dic2 = {'params':[], 'weight_decay':weight_decay, 'lr':max_lr*lr_decay**(num_encoder_layers - 1 - layer)}
        param_lr_weight_decay_dict[key1] = dic1
        param_lr_weight_decay_dict[key2] = dic2
    param_lr_weight_decay_dict['nwd'] = {'params':[], 'weight_decay':0., 'lr':max_lr}
    param_lr_weight_decay_dict['wd'] = {'params':[], 'weight_decay':weight_decay, 'lr':max_lr}

    for name, param in params_optimization:
        if not name.startswith('bert'):
            if not any(nd in name for nd in no_weight_decay):
                param_lr_weight_decay_dict['wd']['params'].append(param)
            else:
                param_lr_weight_decay_dict['nwd']['params'].append(param)
        else:
            if 'layer' not in name or 'layer.0.' in name:
                if not any(nd in name for nd in no_weight_decay):
                    param_lr_weight_decay_dict['layer.0-wd']['params'].append(param)
                else:
                    param_lr_weight_decay_dict['layer.0-nwd']['params'].append(param)
            else:
                layer_name = re.search(r'(layer\.[0-9]{1,2})\.', name).group(1)
                if not any(nd in name for nd in no_weight_decay):
                    param_lr_weight_decay_dict[f'{layer_name}-wd']['params'].append(param)
                else:
                    param_lr_weight_decay_dict[f'{layer_name}-nwd']['params'].append(param)

    if lmmodel is not None:
        for name, param in lmmodel.named_parameters():
            if 'head' in name:
                if not any(nd in name for nd in no_weight_decay):
                    param_lr_weight_decay_dict['wd']['params'].append(param)
                else:
                    param_lr_weight_decay_dict['nwd']['params'].append(param)
            else:
                continue

    optimizer_grouped_parameters = [  param_group for param_group in param_lr_weight_decay_dict.values() if param_group['params'] ]
    return optimizer_grouped_parameters