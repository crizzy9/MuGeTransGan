# Transformer implementation in PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import math, copy, time
# import matplotlib.pyplot as plt
# import seaborn
from helpers import clone_layers

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
        return self.decode(self.encode(src,  src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def deocode(self, memory, src_mask, tgt, tgt_mask):
        return self.deoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


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
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
