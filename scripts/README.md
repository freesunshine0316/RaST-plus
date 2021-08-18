# Scripts for Data Preparing and System Ensemble

1. data preparing

Use ```find_unk.py``` to detect the UNK tokens and the maximum length of any test set.

Use ```gen_folds.py``` to creat 10-fold cross-validation settings for the given training set.

2. ensemble

Use ```convert_generation_to_tags.py``` to convert any outputs from other models into the tagging decisions similar with our RaST model.
It uses the LCS algorithm to align a predicted sentence with the reference, it then gets tags based on the alignments.

Put any number of model predictions in the ``decisions`` format into a folder, make sure the files all take the ``.pred`` suffix.
Finally, use ```python vote_n_merge.py xxx``` to get the ensemble results, where ``xx`` is the folder containing the prediction files.
