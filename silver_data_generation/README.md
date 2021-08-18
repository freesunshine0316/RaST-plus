
Step 1. Run python build_fake_data.py LCCC_LARGE_FILE  to build the base silver data named fake_rewrite_dataset.txt. The LCCC_LARGE_FILE can be found at the LCCC github site https://github.com/thu-coai/CDial-GPT.
This step extracts three turns out of dialogues in LCCC, and creates silver dataset by replacing or dropping phrases in the third turn if they appear in previous turns. POS tagging is used to filter out instances where
the replaced or dropped phrases are verbs or function words.

2. Run python convert_fake_to_json.py to generate fake_rewrite_dataset.txtonlyN.json for the dataset with only nouns for P1 pretraining.
This step creates the json file needed for P1 pretraining, where all data is used to pretrain an LM.

3. Run python create_hard_fake_dataset.py to build P2 pretraining silver data named fake_rewrite_dataset.txtonlyN.hard.json.
This step creates the hard silver dataset used for P2 pretraining, which consists of instances where the third turn after modification is most different from the original third turn.

Dependencies: hanlp, nltk
