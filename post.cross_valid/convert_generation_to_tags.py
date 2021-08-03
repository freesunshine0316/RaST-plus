
import os, sys, json

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

def get_tags(task_input, target):
    task_input_tokens = task_input.split()
    bnd = task_input_tokens.index('|')

    context_tokens = task_input_tokens[:bnd+1]
    tags = [["", "DELETE"], for x in context_tokens]

    source_tokens = task_input_tokens[bnd+1:]
    target_tokens = tokenize(target) + ['*']
    kept_idx = _compute_lcs(source_tokens, target_tokens)
    assert kept_idx[-1] == (len(source_tokens)-1,len(target_tokens)-1)

    for i, (kept_src_idx, kept_tgt_idx) in enumerate(kept_idx):
        tgt_phr_st = 0 if i == 0 else kept_idx[i-1][1] + 1
        tgt_phr_ed = kept_tgt_idx
        tgt_phr = " ".join(target_tokens[tgt_phr_st:tgt_phr_ed])

        last_kept_src_idx = -1 if i == 0 else kept_idx[i-1][0]
        if last_kept_src_idx + 1 == kept_src_idx: # insertion
            tags.append([add_phrase, "KEEP"])
        else: # replacement
            tags.append([add_phrase, "DELETE"])
            for j in range(last_kept_src_idx+2, kept_src_idx):
                tags.apepnd(["", "DELETE"])
            tags.append(["", "KEEP"])
    assert len(task_input_tokens) == len(tags)

    return {'source': task_input_tokens, 'decisions': tags}


def process_file(source_file, target_file):
    with open(source_file, 'r') as f:
        task_inputs = [x.strip().split('\t')[0] for x in f]

    with open(target_file, 'r') as f:
