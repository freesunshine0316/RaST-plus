import json
import nltk
from nltk.metrics import distance
import random

with open('fake_rewrite_dataset.txtonlyN.json') as fh, open('fake_rewrite_dataset.txtonlyN.hard.json', 'w') as wh:
    difficulties = {}
    for line_index, line in enumerate(fh):
        dic = json.loads(line)
        hypo = dic['conv'][2]
        ref = dic['gold_rewrite']
        edit_distance = distance.edit_distance(hypo, ref)
        if edit_distance not in difficulties:
            difficulties[edit_distance] = []
        difficulties[edit_distance].append(line_index)

    saved_instances = []
    for key in difficulties:
        if key >= 25 or key <= 3:
            continue
        else:
            saved_instances.extend(difficulties[key])

    random.seed(1)
    chosen = random.sample(saved_instances, 40000)

    fh.seek(0)
    for line_index, line in enumerate(fh):
        if line_index in chosen:
            wh.write(line)