from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tagging
import utils_data as utils

from typing import Iterable, Mapping, Sequence, Set, Text, Tuple

def tag_to_sequence(tokens, tags):
    tags = [str(t) for t in tags]
    output_tokens = []
    for token, tag in zip(tokens, tags):
        if len(tag.split("|"))>1:
           output_tokens.append(tag.split("|")[1])
        if tag.split("|")[0]=="KEEP":
           output_tokens.append(token)
    return output_tokens

class TaggingConverter(object):
  """Converter from training target texts into tagging format."""

  def __init__(self, phrase_vocabulary, do_swap=True):
    """Initializes an instance of TaggingConverter.

    Args:
      phrase_vocabulary: Iterable of phrase vocabulary items (strings).
      do_swap: Whether to enable the SWAP tag.
    """
    self._phrase_vocabulary = set(
        phrase.lower() for phrase in phrase_vocabulary)
    #additional_phrase = {"肃 宁 县", "甲 醛", "邓 紫 棋 的 歌", "抛 弃 我"}
    #self._phrase_vocabulary|=additional_phrase
    self._do_swap = do_swap
    # Maximum number of tokens in an added phrase (inferred from the
    # vocabulary).
    self._max_added_phrase_length = 0
    # Set of tokens that are part of a phrase in self.phrase_vocabulary.
    self._token_vocabulary = set()
    for phrase in self._phrase_vocabulary:
      tokens = utils.get_token_list(phrase)
      self._token_vocabulary |= set(tokens)
      if len(tokens) > self._max_added_phrase_length:
        self._max_added_phrase_length = len(tokens)

  def compute_tags(self, task,
                   target):
    """Computes tags needed for converting the source into the target.

    Args:
      task: tagging.EditingTask that specifies the input.
      target: Target text.

    Returns:
      List of tagging.Tag objects. If the source couldn't be converted into the
      target via tagging, returns an empty list.
    """
    target_tokens = utils.get_token_list(target.lower())
    tags = self._compute_tags_fixed_order(task.source_tokens, target_tokens)
    # If conversion fails, try to obtain the target after swapping the source
    # order.
    if not tags and len(task.sources) == 2 and self._do_swap:
      swapped_task = tagging.EditingTask(task.sources[::-1])
      tags = self._compute_tags_fixed_order(swapped_task.source_tokens,
                                            target_tokens)
      if tags:
        tags = (tags[swapped_task.first_tokens[1]:] +
                tags[:swapped_task.first_tokens[1]])
        # We assume that the last token (typically a period) is never deleted,
        # so we can overwrite the tag_type with SWAP (which keeps the token,
        # moving it and the sentence it's part of to the end).
        tags[task.first_tokens[1] - 1].tag_type = tagging.TagType.SWAP
    #if not tags:
    #   print("source tokens:", task.source_tokens)
    #   print("target tokens:", target_tokens)
    return tags

  def _compute_tags_fixed_order(self, source_tokens, target_tokens):
    """Computes tags when the order of sources is fixed.

    Args:
      source_tokens: List of source tokens.
      target_tokens: List of tokens to be obtained via edit operations.

    Returns:
      List of tagging.Tag objects. If the source couldn't be converted into the
      target via tagging, returns an empty list.
    """
    source_tokens.append("*")
    target_tokens.append("*")
    
    context = " ".join(source_tokens).split(" [CI] ")[0]
    source_tokens = " ".join(source_tokens).split(" [CI] ")[1]
    context = context.split()
    context.append("[CI]")
    source_tokens = source_tokens.split()
    if source_tokens == ["*"]:
       source_tokens = target_tokens + ["*"]
    
    tags_context = [tagging.Tag('DELETE') for _ in context]
    tags = [tagging.Tag('DELETE') for _ in source_tokens]
    
    # Indices of the tokens currently being processed.
    source_token_idx = 0
    target_token_idx = 0
    while target_token_idx < len(target_tokens):
      tags[source_token_idx], target_token_idx = self._compute_single_tag(
          source_tokens[source_token_idx], target_token_idx, target_tokens)
      # If we're adding a phrase and the previous source token(s) were deleted,
      # we could add the phrase before a previously deleted token and still get
      # the same realized output. For example:
      #    [DELETE, DELETE, KEEP|"what is"]
      # and
      #    [DELETE|"what is", DELETE, KEEP]
      # Would yield the same realized output. Experimentally, we noticed that
      # the model works better / the learning task becomes easier when phrases
      # are always added before the first deleted token. Also note that in the
      # current implementation, this way of moving the added phrase backward is
      # the only way a DELETE tag can have an added phrase, so sequences like
      # [DELETE|"What", DELETE|"is"] will never be created.
      
      if tags[source_token_idx].added_phrase:
        first_deletion_idx = self._find_first_deletion_idx(
            source_token_idx, tags)
        if first_deletion_idx != source_token_idx:
          tags[first_deletion_idx].added_phrase = (
              tags[source_token_idx].added_phrase)
          tags[source_token_idx].added_phrase = ''
      
      source_token_idx += 1
      if source_token_idx >= len(tags):
        break
    target_seq = tag_to_sequence(source_tokens, tags)
    
    if " ".join(target_seq).lower() != " ".join(target_tokens).lower():
       print("source:", source_tokens)
       print("tags:", [str(t) for t in tags])
       print("tag to target sequence:", target_seq)
       print("target tokens:", target_tokens)
    
    #assert " ".join(target_seq).lower() == " ".join(target_tokens).lower()
    # If all target tokens have been consumed, we have found a conversion and
    # can return the tags. Note that if there are remaining source tokens, they
    # are already marked deleted when initializing the tag list.
    if target_token_idx >= len(target_tokens):
       tags_context.extend(tags)
       return tags_context
       #return tags
    if True:
       print("source:", source_tokens)
       print("tags:", [str(t) for t in tags])
       print("tag to target sequence:", target_seq)
       print("target tokens:", target_tokens)

    return []

  def _compute_single_tag(
      self, source_token, target_token_idx,
      target_tokens):
    """Computes a single tag.

    The tag may match multiple target tokens (via tag.added_phrase) so we return
    the next unmatched target token.

    Args:
      source_token: The token to be tagged.
      target_token_idx: Index of the current target tag.
      target_tokens: List of all target tokens.

    Returns:
      A tuple with (1) the computed tag and (2) the next target_token_idx.
    """
    source_token = source_token.lower()
    target_token = target_tokens[target_token_idx].lower()
    if source_token == target_token:
      return tagging.Tag('KEEP'), target_token_idx + 1

    added_phrase = ''
    for num_added_tokens in range(1, self._max_added_phrase_length + 1):
      if target_token not in self._token_vocabulary:
        break
      added_phrase += (' ' if added_phrase else '') + target_token
      next_target_token_idx = target_token_idx + num_added_tokens
      if next_target_token_idx >= len(target_tokens):
        break
      target_token = target_tokens[next_target_token_idx].lower()
      if (source_token == target_token and added_phrase in self._phrase_vocabulary):
        return tagging.Tag('KEEP|' + added_phrase), next_target_token_idx + 1
      #if (source_token == target_token and target_token == target_tokens[-2]):
      #  return tagging.Tag('KEEP|' + added_phrase), next_target_token_idx + 1
      if (source_token == target_token and source_token=="*"):
        return tagging.Tag('KEEP|' + added_phrase), next_target_token_idx + 1
    return tagging.Tag('DELETE'), target_token_idx

  def _find_first_deletion_idx(self, source_token_idx, tags):
    """Finds the start index of a span of deleted tokens.

    If `source_token_idx` is preceded by a span of deleted tokens, finds the
    start index of the span. Otherwise, returns `source_token_idx`.

    Args:
      source_token_idx: Index of the current source token.
      tags: List of tags.

    Returns:
      The index of the first deleted token preceding `source_token_idx` or
      `source_token_idx` if there are no deleted tokens right before it.
    """
    # Backtrack until the beginning of the tag sequence.
    for idx in range(source_token_idx, 0, -1):
      if tags[idx - 1].tag_type != tagging.TagType.DELETE:
        return idx
    return 0


def get_phrase_vocabulary_from_label_map(
    label_map):
  """Extract the set of all phrases from label map.

  Args:
    label_map: Mapping from tags to tag IDs.

  Returns:
    Set of all phrases appearing in the label map.
  """
  phrase_vocabulary = set()
  for label in label_map.keys():
    tag = tagging.Tag(label)
    if tag.added_phrase:
      phrase_vocabulary.add(tag.added_phrase)
  return phrase_vocabulary
