from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections

from bert import tokenization
import tagging
import tagging_converter
from typing import Mapping, MutableSequence, Optional, Sequence, Text
import utils_data as utils 

class BertExample(object):
  """Class for training and inference examples for BERT.

  Attributes:
    editing_task: The EditingTask from which this example was created. Needed
      when realizing labels predicted for this example.
    features: Feature dictionary.
  """

  def __init__(self, input_tokens,
               labels,
               label_action,
               label_start,
               label_end,
               can_convert,
               task, default_label):
    input_len = len(" ".join(input_tokens).split("\t")[0].split())
    if not (input_len == len(label_action) and input_len == len(label_start) and
            input_len == len(label_end) and input_len == len(labels)):
      print("input:", input_tokens)
      print("target:", " ".join(input_tokens).split("\t")[1])
      print("input len:", input_len)
      print("len label_start:", len(label_start))
      raise ValueError(
          'All feature lists should have the same length ({})'.format(
              input_len))

    self.features = collections.OrderedDict([
        ('input_tokens', input_tokens),
        ('labels', labels),
        ('label_action', label_action),
        ('label_start', label_start),
        ('label_end', label_end),
        ('can_convert', can_convert),
    ])
    self.editing_task = task
    self._default_label = default_label


class BertExampleBuilder(object):
  """Builder class for BertExample objects."""

  def __init__(self, label_map, vocab_file,
               max_seq_length, do_lower_case,
               converter):
    """Initializes an instance of BertExampleBuilder.

    Args:
      label_map: Mapping from tags to tag IDs.
      vocab_file: Path to BERT vocabulary file.
      max_seq_length: Maximum sequence length.
      do_lower_case: Whether to lower case the input text. Should be True for
        uncased models and False for cased models.
      converter: Converter from text targets to tags.
    """
    self._label_map = label_map
    self._tokenizer = tokenization.FullTokenizer(vocab_file,
                                                 do_lower_case=do_lower_case)
    self._max_seq_length = max_seq_length
    self._converter = converter
    self._pad_id = self._get_pad_id()
    self._keep_tag_id = self._label_map['KEEP']

  def build_bert_example(
      self,
      sources,
      target = None,
      use_arbitrary_target_ids_for_infeasible_examples = False
  ):
    """Constructs a BERT Example.

    Args:
      sources: List of source texts.
      target: Target text or None when building an example during inference.
      use_arbitrary_target_ids_for_infeasible_examples: Whether to build an
        example with arbitrary target ids even if the target can't be obtained
        via tagging.

    Returns:
      BertExample, or None if the conversion from text to tags was infeasible
      and use_arbitrary_target_ids_for_infeasible_examples == False.
    """
    # Compute target labels.
    if target == None:
       sources = [sources[0]+" <T>"]
    task = tagging.EditingTask(sources)
  
    if target is not None:
      tags = self._converter.compute_tags(task, target)
      if not tags:
        if use_arbitrary_target_ids_for_infeasible_examples:
          # Create a tag sequence [KEEP, DELETE, KEEP, DELETE, ...] which is
          # unlikely to be predicted by chance.
          tags = [tagging.Tag('KEEP') if i % 2 == 0 else tagging.Tag('DELETE')
                  for i, _ in enumerate(task.source_tokens)]
        else:
          return None
    else:
      # If target is not provided, we set all target labels to KEEP.
      tags = [tagging.Tag('KEEP') for _ in task.source_tokens]

    label_action = []
    start = []
    end = []
    labels_ = []
    can_convert = True
    for t in tags:
      t = str(t)
      if len(t.split("|"))>1:
        phrase = t.split("|")[1]
        s_ind, phrase = utils.find_phrase_idx(" ".join(task.source_tokens).lower(), phrase.lower())
        if s_ind==-1:
          start.append(-1)
          end.append(-1)
          can_convert = False
        else:
          start.append(s_ind)
          end.append(s_ind+len(phrase)-1)
      else:
        start.append(-1)
        end.append(-1)
      if t.split("|")[0]=="KEEP":
        label_action.append(0)
      elif t.split("|")[0]=="DELETE":
        label_action.append(1)
      else:
        label_action.append(2)
      labels_.append(t.split("|")[0])

    label_start = []
    label_end = []
    for i in range(len(start)):
      if start[i] == -1 or end[i] == -1:
        label_start.append(start[i])
        label_end.append(end[i])
      else:
        label_start.append(task.char_to_word_offset[int(start[i])])
        label_end.append(task.char_to_word_offset[int(end[i])])
    
    labels=[]
    for i in range(len(labels_)):
        write = labels_[i]+"|"+str(label_start[i])+"#"+str(label_end[i])
        labels.append(write)
    example = BertExample(
        input_tokens=task.source_tokens+["\t"]+target.split(),
        labels=labels,
        label_action=label_action,
        label_start=label_start,
        label_end=label_end,
        can_convert=can_convert,
        task=task,
        default_label=self._keep_tag_id)
    #example.pad_to_max_length(self._max_seq_length, self._pad_id)
    return example

  def _get_pad_id(self):
    """Returns the ID of the [PAD] token (or 0 if it's not in the vocab)."""
    try:
      return self._tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
    except KeyError:
      return 0
