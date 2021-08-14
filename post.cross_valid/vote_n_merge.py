
import os, sys, json
from os import listdir
from os.path import isfile, join
from collections import defaultdict
from transformers import BertTokenizer

def load_file(path):
    print('loading from {}'.format(path))
    with open(path, 'r') as f:
        data = []
        for i, line in enumerate(f):
            line = json.loads(line.strip())
            #if line['source'][-1] != '*':
            #    print('Adding missing * for line {}'.format(i))
            #    line['source'] += ['*']
            #    if len(line['decisions'][0]) == 2:
            #        line['decisions'] += [["", "KEEP"]]
            #    else:
            #        line['decisions'] += [["KEEP", 1.0, 0, 0, 1.0]]
            data.append(line)
    return data

def best_choice(dct):
    dct = sorted(list(dct.items()), key=lambda x: (x[1], len(x[0].split()), x[0]))
    dct = list(reversed(dct))
    if len(dct) == 1 or dct[0][1] > dct[1][1]:
        return dct[0][0]
    if dct[0][0] == 'KEEP':
        return dct[0][0]
    equal_dct = [x[0] for x in dct if x[1] == dct[0][1]]
    equal_dct = sorted(equal_dct, key=lambda x: len(x))
    print(equal_dct)
    return equal_dct[0]

def vote_n_merge(folder):
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large', do_lower_case=False)
    special_tokens = set([tokenizer.cls_token, tokenizer.sep_token,
        tokenizer.unk_token, tokenizer.pad_token, '*', '|'])

    all_data = []
    all_path = []
    for path in listdir(folder):
        path = join(folder, path)
        if isfile(path) and path.endswith('.pred'):
            all_data.append(load_file(path))
            all_path.append(path)

    all_results = []
    for i in range(len(all_data[0])):
        cur_results = []
        cur_source = all_data[0][i]['source']
        inconsis_list = []
        for n in range(len(all_data)):
            if cur_source != all_data[n][i]['source']:
                print(i)
                print(all_path[n])
                print(" ".join(cur_source))
                print(" ".join(all_data[n][i]['source']))
                print("=====")
                inconsis_list.append(n)
        for j in range(len(all_data[0][i]['decisions'])):
            actions = defaultdict(float)
            spans = defaultdict(float)
            for n in range(len(all_data)):
                if n in inconsis_list:
                    continue
                xxx = all_data[n][i]['decisions'][j]
                if len(xxx) == 5:
                    act, act_prob, st, ed, span_prob = xxx
                    if ed > 0 and ed >= st:
                        add_phrase = " ".join(cur_source[st:ed+1])
                    else:
                        add_phrase = ""
                        # separate (0,0) from invlid span, where the prob is considered as '1-span_prob'
                        if (st,ed) != (0,0):
                            span_prob = 1.0 - span_prob
                elif len(xxx) == 2:
                    add_phrase, act = xxx
                    act_prob, span_prob = 1.0, 1.0
                else:
                    assert False
                actions[act] += act_prob
                spans[add_phrase] += span_prob
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

    recover_ori_case(all_results)
    with open(join(folder, 'voted_final.txt'), 'w') as f:
        for x in all_results:
            f.write(x+'\n')

def recover_ori_case(all_results):
    ori_context = [x.strip() for x in open('coai.validation.sentences.txt_2', 'r')]
    assert len(ori_context) == len(all_results)

    for i, (ori_c, rst) in enumerate(zip(ori_context, all_results)):
        ori_c_lower = ori_c.lower()
        new_rst = []
        for x in rst.split():
            if x.isascii() and x.isalpha():
                idx = ori_c_lower.find(x)
                if idx >= 0:
                    new_x = ori_c[idx:idx+len(x)]
                else:
                    new_x = x
                if new_x != x:
                    print('{} ---> {}'.format(x, new_x))
                new_rst.append(new_x)
            else:
                new_rst.append(x)
        all_results[i] = " ".join(new_rst)

vote_n_merge(sys.argv[1])
