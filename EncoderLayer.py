import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from FeedForward import FeedForward
from LayerNorm import LayerNorm
from MultiheadAttention import MultiheadAttention


class EncoderLayer(nn.Module):
  def __init__(self, d_model, n_heads, dropout = 0.1):
    super().__init__()
    self.d_model = d_model
    self.norm_1 = LayerNorm(d_model)
    self.norm_2 = LayerNorm(d_model)
    self.multi_head_attention = MultiheadAttention(n_heads, d_model)
    self.feed_forward = FeedForward(d_model)
    self.dropout_1 = nn.Dropout(dropout) 
    self.dropout_2 = nn.Dropout(dropout)

  def forward(self, x, mask):
    x = x + self.dropout_1(self.multi_head_attention(x, x, x, mask))
    x = self.norm_1(x)

    x = x + self.dropout_2(self.feed_forward(x))
    x = self.norm_2(x)

    return x
