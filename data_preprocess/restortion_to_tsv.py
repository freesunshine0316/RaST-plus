
import os, sys, json
from collections import defaultdict

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


def process(part):
    data = defaultdict(list)
    for i, line in enumerate(open('emnlp/{}.txt'.format(part), 'r')):
        sents = line.strip().split('\t')
        context = ''.join(sents[:5])
        if len(sents[4]) == 0:
            print(line)
            sents[4] = sents[7]
        sents = [' '.join(tokenize(x)) for x in sents]
        con = ' [SEP] '.join(sents[:4])
        inp = sents[4]
        ref = sents[7] if int(sents[6]) == 1 else sents[4]
        data[context].append([con, inp, ref, i])
    print(sum(len(x) for x in data.values()))

    f = open('emnlp.{}.tsv'.format(part), 'w')
    for i, line in enumerate(open('emnlp.{}.csrl'.format(part), 'r')):
        try:
            context = json.loads(line.strip())['sent']
        except:
            print(i)
            continue
        context = ''.join(x for x in context.split() if x not in ('human', 'agent'))
        assert context in data, context
        con, inp, ref, j = data[context][0]
        data[context].pop(0)
        f.write('{} [CI] {}\t{}\n'.format(con, inp, ref))
    print(i+1)
    print(sum(len(x) for x in data.values()))
    f.close()

process('train')
process('dev')
process('test')

