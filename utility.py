import torch
import torch.nn as nn
import copy

# Build a convenient cloning function that can generate multiple encoder or decoder layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])