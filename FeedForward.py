import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from embedding import scaled_dot_product_attention


class FeedForward(nn.Module):
  def __init__(self, d_model, d_ff = 2048, dropout = 0.1):
    super().__init__()
    self.linear_1 = nn.Linear(in_features = d_model, out_features = d_ff)
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(in_features = d_ff, out_features = d_model)

  def forward(self, x):
    return self.linear_2(self.dropout(F.relu(self.linear_1(x))))
