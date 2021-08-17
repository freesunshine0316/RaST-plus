
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


def is_chunked_prefix(full, cur):
    return len(full) > len(cur) and full[:len(cur)] == cur


def restore_unchanged(ori_source, cur_data):
    decisions = []
    last_turn = False
    for x in ori_source:
        if last_turn == False:
            decisions.append(['DELETE', 1.0, 0, 0, 1.0])
        else:
            decisions.append(['KEEP', 0.6, 0, 0, 0.6]) # make it not very certain
        if x == '|':
            last_turn = True # | does not belong to the last turn
    cur_data['source'] = ori_source[:]
    cur_data['decisions'] = decisions

def vote_n_merge(task_inputs, folder):
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large', do_lower_case=False)
    special_tokens = set([tokenizer.cls_token, tokenizer.sep_token,
        tokenizer.unk_token, tokenizer.pad_token, '*', '|'])

    all_files = sorted(listdir(folder), reverse=True) # make sure to process kun's files first

    all_data = []
    all_path = []
    for path in all_files:
        path = join(folder, path)
        if isfile(path) and path.endswith('.pred'):
            all_data.append(load_file(path))
            all_path.append(path)

    all_results = []
    for i in range(len(all_data[0])):
        cur_results = []
        cur_source = all_data[0][i]['source'] # task_inputs[i] #
        inconsis_list = []
        for n in range(len(all_data)):
            if is_chunked_prefix(cur_source, all_data[n][i]['source']):
                print(i)
                print('restore unchanged decisions for {}'.format(all_path[n]))
                print("=====")
                restore_unchanged(cur_source, all_data[n][i])
            if cur_source != all_data[n][i]['source']: # make sure it's consist after restortion
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

def recover_ori_case_kun(all_results):
    ori_context = [x.strip() for x in open('coai.test_with_noise.sentences.txt', 'r')]


def recover_ori_case(all_results):
    ori_context = [x.strip() for x in open('coai.test_with_noise.sentences.txt', 'r')]
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

#print('Loading original source')
#
#unk_words = json.load(open('../unk.json', 'r'))
#unk_mapping = {x:'[unused{}]'.format(i+1) for i, x in enumerate(unk_words)}
#unk_mapping_rev = {'[unused{}]'.format(i+1):x for i, x in enumerate(unk_words)}
#unk_placeholders = list(unk_mapping_rev.keys())
#
#tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
#tokenizer.add_special_tokens({"additional_special_tokens": unk_placeholders})

task_inputs = None
#with open('coai.test_with_noise.sentences.txt', 'r') as f:
#    task_inputs = []
#    for line in f:
#        rst = ["[CLS]",] + line.strip().split()
#        rst = [unk_mapping.get(x,x) for x in rst]
#        rst = list(map(tokenizer.tokenize, rst))
#        rst = [item for sub_tokens in rst for item in sub_tokens]
#        rst = [unk_mapping_rev.get(x,x) for x in rst]
#        task_inputs.append(rst)

vote_n_merge(task_inputs, sys.argv[1])
