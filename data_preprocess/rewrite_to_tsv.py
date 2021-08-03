
import os, sys, json

def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True

def tokenize(sen):
    result = []
    english_token = []
    tokens = list(sen)
    for i in range(len(tokens)):
        if is_all_chinese(tokens[i]):
            if len(english_token) > 0:
                result.append("".join(english_token))
                english_token = []
            result.append(tokens[i])
        else:
            english_token.append(tokens[i])
    if len(english_token)>0:
        result.append("".join(english_token))
    return result


f = open('acl.corpus.tsv', 'w')
for i, line in enumerate(open('acl/acl.corpus', 'r')):
    sents = line.strip().split('\t\t')
    sents = [' '.join(tokenize(x)) for x in sents]
    c1, c2, inp, ref = sents
    f.write('{} [SEP] {} [CI] {}\t{}\n'.format(c1, c2, inp, ref))
f.close()

os.system('head -n 18000 acl.corpus.tsv > acl.corpus.train.tsv')
os.system('tail -n 2000 acl.corpus.tsv > tmp')
os.system('head -n 1000 tmp > acl.corpus.dev.tsv; tail -n 1000 tmp > acl.corpus.test.tsv')
os.system('rm tmp acl.corpus.tsv')

