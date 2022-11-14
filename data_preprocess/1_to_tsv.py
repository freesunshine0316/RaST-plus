import os, json


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
    if len(english_token) > 0:
        result.append("".join(english_token))
    return result


def process_noref(part):
    f = open('{}.sentences.txt'.format(part), 'w')
    for i, line in enumerate(open('{}.txt'.format(part), 'r')):
        line = json.loads(line.strip())
        c1, c2, inp = ' '.join(tokenize(line['query-01'])), ' '.join(tokenize(line['response-01'])), \
                ' '.join(tokenize(line['query-02']))
        context = '{} [SEP] {}'.format(c1, c2)
        f.write('{} | {} *\n'.format(context, inp))
    f.close()


def process(part, every_n):
    if every_n == 0:
        fpos = open(f'{part}_pos.tsv', 'w')
        fneg = open(f'{part}_neg.tsv', 'w')
    else:
        fpos = fneg = open(f'{part}_every_{every_n}.tsv', 'w')

    last_context = []
    for i, line in enumerate(open(f'{part}.txt', 'r')):
        line = json.loads(line.strip())
        c1, c2, inp, ref = ' '.join(tokenize(line['query-01'])), ' '.join(tokenize(line['response-01'])), \
                ' '.join(tokenize(line['query-02'])), 'Unknown'
        if 'query-02-rewrite' in line:
            ref = ' '.join(tokenize(line['query-02-rewrite']))
        fpos.write(f'{c1} [SEP] {c2} [CI] {inp}\t{ref}\n')
        if i > 0 and (every_n == 0 or i % every_n == 0):
            fneg.write(f'{last_context[0]} [SEP] {last_context[1]} [CI] {c1}\t{c1}\n')
        last_context = [c1, c2]
    fpos.close()
    fneg.close()


os.system('head -n 19000 train.txt > coai.train.txt')
os.system('tail -n 1000 train.txt > coai.dev.txt')
process('coai.dev', every_n=0)
for every_n in [1, 3, 5]:
    process('coai.train', every_n=every_n)
#process_noref('validation')
#process_noref('test_with_noise')
#os.system('mv validation.sentences.txt coai.validation.sentences.txt')
#os.system('mv test_with_noise.sentences.txt coai.test_with_noise.sentences.txt')
