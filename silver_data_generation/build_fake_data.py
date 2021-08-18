import json
import zipfile
from generate_shortened_version import generate_short
import hanlp
import re
import sys

LCCC_folder = sys.argv[1]  # 'LCCC-large.zip'

KEEP_POS_SET = {'VA', 'VV', 'NN', 'NT', 'NR', 'JJ', 'FW'}
predictor = hanlp.load('CTB9_POS_ALBERT_BASE', devices=['cpu'], verbose=False, tasks='pos', skip_tasks='tok*')
predictor('今天天气好。')

with zipfile.ZipFile(LCCC_folder) as zpfh, open('fake_rewrite_dataset.txt', 'w') as wfh:
    with zpfh.open('LCCD.json') as fh:
        convs = json.load(fh)
        for conv in convs:
            first_three_turns = [re.sub('\s+', ' ', x) for x in conv[:3]]
            if len(first_three_turns) < 3:
                continue
            vocab_set = set()
            for sent in first_three_turns[:2]:
                vocab_set.update(sent.split(' '))
            final_sent = first_three_turns[2].split(' ')
            if len(''.join(final_sent)) >= 100:
                continue
            candidate_modification_words = []
            parts_of_speech = predictor([final_sent])[0]
            for word_index, word in enumerate(final_sent):
                if word in vocab_set and parts_of_speech[word_index] in KEEP_POS_SET:
                    candidate_modification_words.append(word)
                else:
                    candidate_modification_words.append('')
            if all([x == '' for x in candidate_modification_words]):
                continue

            short_sent = generate_short(final_sent, candidate_modification_words, parts_of_speech)
            print(first_three_turns[0], file=wfh)
            print(first_three_turns[1], file=wfh)
            print(first_three_turns[2], file=wfh)
            print(short_sent, file=wfh)
            print(candidate_modification_words, file=wfh)
            print(vocab_set, file=wfh)

            print('', file=wfh)
