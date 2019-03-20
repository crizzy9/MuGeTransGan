# Transformer implementation in PyTorch
import  math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from modules import clone_layers

# Implementation of Transformer as per Attention is all you need paper.

# Base Encoder Decoder model
class EncoderDecoder(nn.Module):
    """
    Standard Encoder-Decoder Architecture. Base class for the transformer model
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Process masked source and target sequences
        """
        print("ed forward")
        return self.decode(self.encode(src,  src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        print("ed enc")
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        print("ed dec")
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    Define linear and softmax layer for generation
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.projection = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.projection(x), dim=-1)


# Could be replaced with nn.LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2


# LayerNorm(x+SubLayer(x))
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a LayerNorm
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size) # Could use nn.LayerNorm
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # Apply residual connection to a sublayer with the same size
        return x + self.dropout(sublayer(self.norm(x))) # why take normalized sublayer when encoder layers are already normalized?



class Encoder(nn.Module):
    """
    Create encoder with a stack of `num_layers`
    """
    def __init__(self, layer, num_layers=1):
        super(Encoder, self).__init__()
        self.layers = clone_layers(layer, num_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Process input and mask through all the layers
        """
        print("e for")
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    Self-attentive feed forward encoder
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # Each layer has two sub-layers. The first is a multi-head
        # self-attention mechanism, and the second is a simple, position-wise
        # fully connected feed- forward network.
        self.sublayer = clone_layers(SublayerConnection(size, dropout), 2)
    
    def forward(self, x, mask):
        """
        input goes through self attention layer and then a feed forward network
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        print("enc layer", x)
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """
    Create decoder with a stack of `num_layers` with masking
    """
    def __init__(self, layer, num_layers=1):
        super(Decoder, self).__init__()
        self.layers = clone_layers(layer, num_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        print("d for")
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Self-attentive and Source(Encoder) Attentive feed forward decoder
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # Each layer has three sub-layers. The first is a multi-head
        # self-attention mechanism, and the second is a source attention layer with all the layers of the encoder
        # and the third is a simple, position-wise fully connected feed- forward network.
        self.sublayer = clone_layers(SublayerConnection(size, dropout), 3)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        """
        input goes through self attention layer and then a source attention layer and then finally a feed forward network
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        print("dec layer", x)
        return self.sublayer[2](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    """
    Implement Feed Forward Network in each layer of encoder and decoder stack
    with 2 linear layers and a relu in between
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


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

    # @staticmethod
    def scaled_dot_product_attention(self, query, key, value, mask=None, dropout=None):
        """
        Computes Scaled Dot product attention based on given Q, K and V
        The output is a weighted sum of the values where the weights are determined
        by a compatibility function of the query with the corresponding key
        """
        d_k = query.size(-1)
        print("sdpa", d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # print(scores.size())
        # scores.unsqueeze_(1)
        # print(scores.size())
        if mask is not None:
            # print(mask.size())
            scores = scores.masked_fill_(mask == 0, -1e9) # ??? -1e9

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not  None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

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
                for l, x in zip(self.linears, (query, key, value))]

        # Apply attention on all projected vectors in the batch
        x, self.attn = self.scaled_dot_product_attention(query, key, value, mask=mask, dropout=self.dropout)

        # Concatenate and apply final layer
        # Why concat on x?
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h, self.d_k)
        # print(x.size())
        x = x.view(1, x.size(1), -1)
        # print(x.size())
        print("ma for")

        return self.linears[-1](x)


class Embeddings(nn.Module):
    """
    Convert input/output tokens to learned embeddings(vector of dimension d_model)
    and feed it to the model. Take softmax of the output to get the prediction.
    The same weight matrix is shared between the two embedding layers
    and the pre-softmax linear transformation??
    """
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # in embedding layer weights are multiplied by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


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
