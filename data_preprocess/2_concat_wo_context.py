import os, sys, json


def process(inpaths, outpath):
    f = open(outpath, 'w')
    for path in inpaths:
        for line in open(path, 'r'):
            x = line.split(' [CI] ')[1]
            f.write(x)
    f.close()


inpaths = ['coai.{}.tsv'.format(x) for x in ['train_every_1', 'train_every_3', 'train_every_5', 'dev_pos', 'dev_neg']]
outpath = 'all.wo_context.tsv'
process(inpaths, outpath)

#inpaths = ['pretrain_hard.{}.tsv'.format(x) for x in ['train', ]]
#outpath = 'pretrain_hard.wo_context.tsv'
#process(inpaths, outpath)
