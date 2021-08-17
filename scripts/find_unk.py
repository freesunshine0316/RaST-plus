
import os, sys, json
from transformers import BertTokenizer


def check_file(tokenizer, path):
    unk_set = set()
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split()
            tokens = []
            for x in line:
                if x not in unk_set:
                    rst = tokenizer.tokenize(x)
                    tokens.extend(rst)
                    if tokenizer.unk_token in rst:
                        unk_set.add(x)
            if len(tokens) > 175:
                print('!!! Line {} with {} tokens'.format(i, len(tokens)))
    print('{} {} '.format(list(unk_set), len(unk_set)))

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
check_file(tokenizer, 'coai.test_with_noise.sentences.txt')
