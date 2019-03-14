import torch.nn as nn

def clone_layers(module, num_layers):
    """
    Replicate `num_layers` of type module
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])

