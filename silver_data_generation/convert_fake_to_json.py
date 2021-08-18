import json

fake_dataset = 'fake_rewrite_dataset.txt'
POSSIBLE_EVENTS_PRONOUNS = ['那个', '这个', '那样', '这样']

def get_words_sent(sent_with_space):
    words = sent_with_space.strip().split(' ')
    sent = ''.join(words)
    return words, sent

with open(fake_dataset) as fh, open(fake_dataset+'.json', 'w') as wfh, open(fake_dataset+'onlyN.json', 'w') as wnfh:

    for line_index, line in enumerate(fh):
        if line_index % 7 == 0:
            entry = {}
            words, sent = get_words_sent(line)
            sentences = [sent]
            sent_toks = [words]
        elif line_index % 7 == 1:
            words, sent = get_words_sent(line)
            sentences.append(sent)
            sent_toks.append(words)
        elif line_index % 7 == 2:
            words, sent = get_words_sent(line)
            entry['gold_rewrite'] = sent
            entry['gold_rewrite_toks'] = words
        elif line_index % 7 == 3:
            query = eval(line.strip())
            if len(query) != len(entry['gold_rewrite_toks']):
                entry['rewrite_type'] = 'N'
            else:
                for word_index, word in enumerate(query):
                    if word != entry['gold_rewrite_toks'][word_index] and word in POSSIBLE_EVENTS_PRONOUNS:
                        entry['rewrite_type'] = 'V'
                        break
                else:
                    entry['rewrite_type'] = 'X'
            sentences.append(''.join(query))
            sent_toks.append(query)
            entry['conv'] = sentences
            entry['conv_toks'] = sent_toks
        elif line_index % 7 == 4 or line_index % 7 == 5:
            pass
        elif line_index % 7 == 6:
            skip_sent = False
            for sent in entry['conv']:
                if not sent.strip():
                    skip_sent = True
            if skip_sent:
                continue
            json_string = json.dumps(entry, ensure_ascii=False)
            # print(entry)
            print(json_string, file=wfh)
            if entry['rewrite_type'] == 'N':
                print(json_string, file=wnfh)