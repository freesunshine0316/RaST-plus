
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

def process_noref(part):
    f = open('{}_inp'.format(part), 'w')
    for i, line in enumerate(open('{}.txt'.format(part), 'r')):
        line = json.loads(line.strip())
        c1, c2, inp = ' '.join(tokenize(line['query-01'])), ' '.join(tokenize(line['response-01'])), \
                ' '.join(tokenize(line['query-02']))
        context = '{} [SEP] {}'.format(c1, c2)
        f.write('{} | {} *\n'.format(context, inp))
    f.close()

def process(part):
    f = open('{}.tsv'.format(part), 'w')
    for i, line in enumerate(open('{}.txt'.format(part), 'r')):
        line = json.loads(line.strip())
        c1, c2, inp, ref = ' '.join(tokenize(line['query-01'])), ' '.join(tokenize(line['response-01'])), \
                ' '.join(tokenize(line['query-02'])), 'Unknown'
        if 'query-02-rewrite' in line:
            ref = ' '.join(tokenize(line['query-02-rewrite']))
        f.write('{} [SEP] {} [CI] {}\t{}\n'.format(c1, c2, inp, ref))
    f.close()

process('train')
process_noref('validation')
os.system('head -n 17000 train.tsv > coai.train.tsv')
os.system('tail -n 3000 train.tsv > coai.dev.tsv')

