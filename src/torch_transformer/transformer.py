# Transformer implementation in PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
import seaborn
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


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()


