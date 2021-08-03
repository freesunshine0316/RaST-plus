
import os, sys, json
from os import listdir
from os.path import isfile, join
from collections import defaultdict
from transformers import BertTokenizer


def load_file(path):
    print('loading from {}'.format(path))
    with open(path, 'r') as f:
        data = [json.loads(line.strip()) for line in f]
    return data

def best_choice(dct):
    dct = sorted(list(dct.items()), key=lambda x: -x[1])
    if len(dct) > 1:
        print(dct)
    return dct[0][0]

def vote_n_merge(folder):
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large', do_lower_case=False)
    special_tokens = set([tokenizer.cls_token, tokenizer.sep_token,
        tokenizer.unk_token, tokenizer.pad_token, '*', '|'])

    all_data = []
    for path in listdir(folder):
        path = join(folder, path)
        if isfile(path) and path.endswith('.pred'):# and \
                #(path.find('fold_1_') >=0 or path.find('fold_2') >=0 or path.find('fold_3') >=0 or path.find('fold_4') >=0):
            all_data.append(load_file(path))

    all_results = []
    for i in range(len(all_data[0])):
        cur_results = []
        cur_source = all_data[0][i]['source']
        for j in range(len(all_data[0][i]['decisions'])):
            actions = defaultdict(float)
            spans = defaultdict(float)
            for n in range(len(all_data)):
                act, act_prob, st, ed, span_prob = all_data[n][i]['decisions'][j]
                actions[act] += act_prob
                if ed > 0 and ed >= st:
                    add_phrase = " ".join(cur_source[st:ed+1])
                    spans[add_phrase] += span_prob
                else: # separate (0,0) from invlid span, where the prob is considered as '1-span_prob'
                    add_phrase = ""
                    spans[add_phrase] += 1.0 if (st,ed) == (0,0) else 1.0 - span_prob
            best_act = best_choice(actions)
            best_phr = best_choice(spans)
            if best_phr != "":
                cur_results.append(best_phr)
            if best_act == 'KEEP':
                cur_results.append(cur_source[j])

        cur_results = " ".join(cur_results).split()
        for tkn in special_tokens:
            while tkn in cur_results:
                cur_results.remove(tkn)

        if len(cur_results)==0:
           cur_results.append("*")
        elif len(cur_results) > 1 and cur_results[-1]=="*":
           cur_results = cur_results[:-1]
        cur_results = " ".join(cur_results).replace(" ##", "").strip()
        all_results.append(cur_results)

    with open(join(folder, 'voted_final.txt'), 'w') as f:
        for x in all_results:
            f.write(x+'\n')

vote_n_merge('10fold_spanthres05')
