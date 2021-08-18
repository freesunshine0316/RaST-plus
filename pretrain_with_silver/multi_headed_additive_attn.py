""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn

from misc import generate_relative_positions_matrix,\
                            relative_matmul
from misc import aeq

class Additive_Attention(nn.Module):
    def __init__(self, model_dim, dropout=0.1):

        super(Additive_Attention, self).__init__()

        self.linear_concat = nn.Linear(model_dim*2,model_dim)
        self.linear_logit = nn.Linear(model_dim,1)

        self.log_softmax = nn.LogSoftmax(dim=-1)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask):
        """ Additive attention mechanism. This layer is implemented using a
            one layer feed forward neural network
        :param queries: A tensor with shape [batch, heads, length_q, depth_k]
        :param keys: A tensor with shape [batch, heads, length_kv, depth_k]
        :param values: A tensor with shape [batch, heads, length_kv, depth_v]
        :param bias: A tensor
        :param concat: A boolean value. If ``concat'' is set to True, then
            the computation of attention mechanism is following $tanh(W[q, k])$.
            When ``concat'' is set to False, the computation is following
            $tanh(Wq + Vk)$
        :param keep_prob: a scalar in [0, 1]
        :param dtype: An optional instance of tf.DType
        :param scope: An optional string, the scope of this layer
        :returns: A dict with the following keys:
            weights: A tensor with shape [batch, length_q, length_kv]
        """
        def attention_bias(inputs, inf=-1e9):
            mask = inputs
            # ret = (1.0 - mask) * inf
            ret = torch.where(mask == 0, inf, 1.)
            ret = ret.unsqueeze(dim=1)
            ret = ret.unsqueeze(dim=1)
            return ret

        bias = attention_bias(mask)

        length_q = queries.size(2)
        length_kv = keys.size(2)

        queries = queries.unsqueeze(dim=3) #[bs, 1, len_q, 1, size]
        keys = keys.unsqueeze(dim=2) # [bs, 1, 1, len_k, size]
        q = queries.repeat(1, 1, 1, length_kv, 1)
        k = keys.repeat(1, 1, length_q, 1, 1)

        combined = torch.tanh(self.linear_concat(torch.cat((q, k), dim=-1)))

        # shape: [batch, heads, length_q, length_kv]
        logits = self.linear_logit(combined).squeeze(-1)

        if bias is not None:
            logits = logits + bias

        weights = self.log_softmax(logits)  #
        # weight_mask = torch.where(mask==0, 1e-9, 1)
        # weights = weights.squeeze(1) * weight_mask.unsqueeze(-1)
        #weights = self.dropout(weights)

        return weights.squeeze(1) # [bs, seq, seq]

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.
    .. mermaid::
       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O
    Also includes several additional tricks.
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1,
                 max_relative_positions=0):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.max_relative_positions = max_relative_positions
        self.additive_attention = Additive_Attention(model_dim, dropout)

        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size, self.dim_per_head)

    def forward(self, key, value, query, mask=None, type=None):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask indicating which keys have
               non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):
           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        assert self.head_count == 1, "We want a single attention distribution, \
                not multiple ones for multiple heads"

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)
        device = key.device

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        key = self.linear_keys(key)
        value = self.linear_values(value)
        query = self.linear_query(query)
        key = shape(key)
        value = shape(value)

        if self.max_relative_positions > 0 and type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions,
                cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix.to(device))
            ##  1 or key_len x key_len x dim_per_head
            #relations_values = self.relative_positions_embeddings(
            #    relative_positions_matrix.to(device))

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        
        # batch x num_heads x query_len x key_len
        #query_key = torch.matmul(query, key.transpose(2, 3))

        #if self.max_relative_positions > 0 and type == "self":
        #    scores = query_key + relative_matmul(query, relations_keys, True)
        #else:
        #    scores = query_key
        #scores = scores.float()

        scores = self.additive_attention(query, key, value, mask)

        #if mask is not None:
        #    mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
        #    scores = scores.masked_fill(mask, -1e18)

        # dist = torch.clamp(scores, 1e-6, 1.0)
        # dist = dist / dist.sum(-1,keepdim=True)
        dist = scores

        return dist.view(batch_size, query_len, key_len)
