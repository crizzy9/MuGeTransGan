import math, copy
import numpy as np
import torch
import torch.nn as nn

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

