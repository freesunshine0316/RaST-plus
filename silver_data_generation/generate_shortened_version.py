import random
POSSIBLE_PRONOUNS = ['他', '她', '它', '那个']
POSSIBLE_EVENTS_PRONOUNS = ['那个', '这个', '那样', '这样']
NOUNS = {'NN', 'NT', 'NR'}
def remove_word(long_sent_words, candidate_word_index, candidate_words, parts_of_speech):
    long_sent_words[candidate_word_index] = ''
    if len(long_sent_words) > candidate_word_index + 2 and (long_sent_words[candidate_word_index+1] == '的' or
    long_sent_words[candidate_word_index+1] == '们'):
        long_sent_words[candidate_word_index + 1] = ''
    if candidate_word_index > 0 and long_sent_words[candidate_word_index-1] == '跟':
        long_sent_words[candidate_word_index - 1] = ''
    if len(long_sent_words) > candidate_word_index + 2 and candidate_words[candidate_word_index + 1] != '':
        remove_word(long_sent_words, candidate_word_index+1, candidate_words, parts_of_speech)

def replace_word(long_sent_words, candidate_word_index, candidate_words, parts_of_speech):
    if parts_of_speech[candidate_word_index] in NOUNS:
        long_sent_words[candidate_word_index] = random.sample(POSSIBLE_PRONOUNS, 1)[0]
    else:
        long_sent_words[candidate_word_index] = random.sample(POSSIBLE_EVENTS_PRONOUNS, 1)[0]
    if len(long_sent_words) > candidate_word_index + 2 and candidate_words[candidate_word_index + 1] != '':
        remove_word(long_sent_words, candidate_word_index+1, candidate_words, parts_of_speech)

def generate_short(long_sent_words, candidate_words, parts_of_speech):
    #  long_sent_words = [赤焰战场, 的, 主演, 跟, 蒂尔达, 一样, 气场逼人]
    # candidate_words = [赤焰战场, '', '', '', 蒂尔达, '', '']

    new_short_sent = long_sent_words[:]
    switch = False

    for word_index in range(len(candidate_words)):
        if candidate_words[word_index] == '':
            switch = False
        elif candidate_words[word_index] != '' and switch is True:
            continue
        elif candidate_words[word_index] != '':
            switch = True
            change = random.random()
            if parts_of_speech[word_index] in NOUNS:
                if change < 0.75:
                    remove_word(new_short_sent, word_index, candidate_words, parts_of_speech)
                elif 0.75 <= change < 0.90:
                    replace_word(new_short_sent, word_index, candidate_words, parts_of_speech)
                else:
                    continue
            else:
                replace_word(new_short_sent, word_index, candidate_words, parts_of_speech)

    new_short_sent = [x for x in new_short_sent if x != '']
    return new_short_sent

# if __name__ == '__main__':
#     long_sent_words = ['就', '感觉', '发际', '线', '很', '茂盛', '啊', '！', '我', '最近', '突然', '发现自己', '发际', '线', '好像', '后移', '了', '哭哭', '…', '…']
#     candidate_words = ['', '', '发际','线', '', '', '', '', '', '', '', '', '发际','线', '', '', '', '', '', '']
#     POS = ['就', '感觉', '发际', '线', '很', '茂盛', '啊', '！', '我', '最近', '突然', '发现自己', '发际', '线', '好像', '后移', '了', '哭哭', '…', '…']
#     print(generate_short(long_sent_words, candidate_words, ))