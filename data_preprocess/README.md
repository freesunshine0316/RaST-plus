# Converting target into tagging sequence
In this part, we provide scripts for how to convert target sequence into tagging sequence. 

## Data Preprocess
This code is customized for the COAI competition, while this model was initially tested on [Rewrite data](https://www.aclweb.org/anthology/P19-1003.pdf) and [Restoration data](https://www.aclweb.org/anthology/D19-1191.pdf) as well.

First, we extract phrases by aligning the source (current utterance) and target (target rewritten utterance). Before doing that, put data into this folder and execute ``python  1_to_tsv.py`` to convert it into the TSV format. In this tsv file, there are two columns, first is the multi-turn dialogue, the second is the rewrite. Next execute ``python  2_concat_wo_context.py`` to get the version where only the last turn and the rewrite are kept.

Run this command line to extract phrases map for the data:
```bash
sh phrase_voc_optimization.sh
```
It basically uses the ``wo_content`` file as input.

Then, we can use the extracted phrases to locate the spans which need to be inserted into current utterance:
```bash
sh convert_target_to_tags.sh
```
It uses the full TSV file as input.

Finally, the preprocessed data files are included in the data_out folder. 
