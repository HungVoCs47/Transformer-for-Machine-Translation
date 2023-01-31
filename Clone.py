import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from embedding import scaled_dot_product_attention


import copy
def clone_layer(module, N):
  return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])
