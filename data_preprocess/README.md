# Converting target into tagging sequence
In this part, we provide scripts for how to convert target sequence into tagging sequence. 

## Data Preprocess
We use [Rewrite data](https://www.aclweb.org/anthology/P19-1003.pdf) and [Restoration data](https://www.aclweb.org/anthology/D19-1191.pdf) in this work.

First, we extract phrases by aligning the source (current utterance) and target (target rewritten utterance). Before doing that, put the data (including train, valid and test) into data/ folder. In this folder, the three files should be concatenated into train_valid_test_wo_context.tsv. In this tsv file, there are two columns, first is the multi-turn dialogue, the second is the rewrite. 

Run this command line to extract phrases map for the data:
```bash
sh phrase_voc_optimization.sh
```

Then, we can use the extracted phrases to locate the spans which need to be inserted into current utterance:
```bash
sh convert_target_to_tags.sh
```

Finally, the preprocessed data files are included in the data_out folder. 
