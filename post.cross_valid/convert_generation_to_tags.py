
import os, sys, json
from transformers import BertTokenizer

unk_words = ['抔', '樨', '亓', '頔', '媺', '郄', '骝', '俣', '琤', '吔', '晧', '罇', '甑', '湉']
unk_mapping = {x:'[unused{}]'.format(i+1) for i, x in enumerate(unk_words)}
unk_mapping_rev = {'[unused{}]'.format(i+1):x for i, x in enumerate(unk_words)}
unk_placeholders = list(unk_mapping_rev.keys())

def _lcs_table(source, target):
    """Returns the Longest Common Subsequence dynamic programming table."""
    rows = len(source)
    cols = len(target)
    lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if source[i - 1] == target[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
    return lcs_table

def _backtrack(table, source, target, i, j):
    if i == 0 or j == 0:
        return []
    if source[i - 1] == target[j - 1]:
        return _backtrack(table, source, target, i - 1, j - 1) + [(i-1,j-1)]
    if table[i][j - 1] > table[i - 1][j]:
        return _backtrack(table, source, target, i, j - 1)
    else:
        return _backtrack(table, source, target, i - 1, j)

def _compute_lcs(source, target):
    """Computes the Longest Common Subsequence (LCS).
    Description of the dynamic programming algorithm:
    https://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Args:
        source: List of source tokens.
        target: List of target tokens.
    Returns:
        List of aligned (source, target) idx pairs in the LCS.
    """
    table = _lcs_table(source, target)
    return _backtrack(table, source, target, len(source), len(target))

def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True

def alphabet_tokenize(sen):
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

def tokenize(string, is_alphabet_splitted, tokenizer):
    if is_alphabet_splitted:
        rst = string.split()
    else:
        rst = alphabet_tokenize(string)
    if tokenizer == None:
        return rst
    rst = [unk_mapping.get(x,x) for x in rst]
    rst = list(map(tokenizer.tokenize, rst))
    rst = [item for sub_tokens in rst for item in sub_tokens]
    rst = [unk_mapping_rev.get(x,x) for x in rst]
    return rst

def get_tags(task_input, target, tokenizer, id):
    task_input_tokens = tokenize(task_input, True, tokenizer)
    bnd = task_input_tokens.index('|')
    context_tokens = task_input_tokens[:bnd+1]
    tags = [["", "DELETE"] for x in context_tokens]

    source_tokens = task_input_tokens[bnd+1:]
    target_tokens = tokenize(target, False, tokenizer) + ['*']
    kept_idx = _compute_lcs(source_tokens, target_tokens)
    assert kept_idx[-1] == (len(source_tokens)-1,len(target_tokens)-1)

    #print(' '.join('{}({})'.format(x,i) for i, x in enumerate(source_tokens)))
    #print(' '.join('{}({})'.format(x,i) for i, x in enumerate(target_tokens)))
    #print(kept_idx)

    for i, (kept_src_idx, kept_tgt_idx) in enumerate(kept_idx):
        tgt_phr_st = 0 if i == 0 else kept_idx[i-1][1] + 1
        tgt_phr_ed = kept_tgt_idx
        tgt_phr = " ".join(target_tokens[tgt_phr_st:tgt_phr_ed])

        last_kept_src_idx = -1 if i == 0 else kept_idx[i-1][0]
        if last_kept_src_idx + 1 == kept_src_idx: # insertion
            tags.append([tgt_phr, "KEEP"])
        else: # replacement
            tags.append([tgt_phr, "DELETE"])
            for j in range(last_kept_src_idx+2, kept_src_idx):
                tags.append(["", "DELETE"])
            tags.append(["", "KEEP"])
    assert len(task_input_tokens) == len(tags)
    #print(tags)
    #print('-----')

    return {'source': task_input_tokens, 'decisions': tags}


def process_file(source_file, target_file):
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
    tokenizer.add_special_tokens({"additional_special_tokens": unk_placeholders})
    with open(source_file, 'r') as f:
        task_inputs = ["[CLS] "+x.strip().split('\t')[0] for x in f]

    with open(target_file, 'r') as f:
        targets = []
        for line in f:
            line = json.loads(line.strip())
            targets.append(line['query-02-rewrite'])

    with open(target_file+'.pred', 'w') as f:
        for i, (inp, tgt) in enumerate(zip(task_inputs, targets)):
            jobj = json.dumps(get_tags(inp, tgt, tokenizer, i), ensure_ascii=False)
            f.write(jobj+'\n')

for i in range(10):
    base = '10fold_kun_v2/v2_fintune_{}_res.txt'.format(i)
    print(base)
    process_file('coai.validation.sentences.txt', base)
