import math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def clone_layers(module, num_layers):
    """
    Replicate `num_layers` of type module
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])

def subsequent_mask(size):
    """
    Mask out subsequent layers of encoder/decoder for self attention
    """
    attn_shape = (1, size, size) # ???
    # np triu returns a triangular matrix with elements below kth diagonal zeroed
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0 # == 0 will interchange 0's and 1's

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    Computes Scaled Dot product attention based on given Q, K and V
    The output is a weighted sum of the values where the weights are determined
    by a compatibility function of the query with the corresponding key
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) # ??? -1e9

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not  None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h: number of heads
        :param d_model: model size
        :param dropout: dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # Assuming d_k always equals d_v
        self.d_k = d_model // h
        self.h = h
        # *d_model because we are doing linear projections from d_model => h x d_k
        self.linears = clone_layers(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implementation for Multi Head Attention using scaled dot product
        """
        if mask is not None:
            # same mask gets applied to all heads
            mask.unsqueeze(1)

        n_batches = query.size(0)

        # performing all the linear projections in batch from d_model => h x d_k 
        # zip will only zip the first 3 linear layers and leave the last
        query, key, value = [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
                for l, x in zip(self.layers, (query, key, value))]

        # Apply attention on all projected vectors in the batch
        x, self.attn = scaled_dot_product_attention(query, key, value, mask=mask, dropout=self.dropout)

        # Concatenate and apply final layer
        # Why concat on x?
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h, self.d_k)
        return self.linears[-1](x)


class Embeddings(nn.Module):
    """
    Convert input/output tokens to learned embeddings(vector of dimension d_model)
    and feed it to the model. Take softmax of the output to get the prediction.
    The same weight matrix is shared between the two embedding layers
    and the pre-softmax linear transformation??
    """
    def __init__(self, d_model, vocab):
        super(PositionwiseFeedForward, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # in embedding layer weights are multiplied by sqrt(d_model)
        self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Implement positional encoding with sinusoids
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        """
        PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        2i represents all even numbers/columns
        2i + 1 represents all the odd numbers/columns
        exp(log(pos/10000^(2i/d_model)))
        = exp(log(pos) - log(10000)*(2i/model))
        = exp(log(pos) + 2i*(-log(10000)/model))
        = pos * exp(2i(-log(10000)/model))
        """
        div_term = torch.exp(torch.arange(0., d_model, 2)*-(math.log(10000.0)/d_model))
        # all even columns 0, 2, 4...
        pe[:, 0::2] = torch.sin(position * div_term)
        # all odd columns 1, 3, 5...
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #  [1, 5000, 512]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Variable deprecated is it needed?
        # x = x + self.pe[:, :x.size(1)]
        # Adding the embeddings with the positional encoding
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class RelativeEncoding(nn.Module):
    """
    Implement relative encoding
    """
    def __init__(self):
        pass
