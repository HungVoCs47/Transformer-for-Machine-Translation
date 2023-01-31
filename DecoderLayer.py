import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from FeedForward import FeedForward
from LayerNorm import LayerNorm
from MultiheadAttention import MultiheadAttention


class DecoderLayer(nn.Module):
  def __init__(self, d_model, n_heads, dropout = 0.1):
    super().__init__()
    self.norm_1 = LayerNorm(d_model)
    self.norm_2 = LayerNorm(d_model)
    self.norm_3 = LayerNorm(d_model)
    self.dropout_1 = nn.Dropout(dropout) 
    self.dropout_2 = nn.Dropout(dropout)
    self.dropout_3 = nn.Dropout(dropout)

    self.multi_head_attention_1 = MultiheadAttention(n_heads, d_model)
    self.multi_head_attention_2 = MultiheadAttention(n_heads, d_model)

    self.feed_forward = FeedForward(d_model)

  def forward(self, x, encoder_output, first_mask, second_mask):
    x = x + self.dropout_1(self.multi_head_attention_1(x, x, x, first_mask))
    x = self.norm_1(x)
    
    x = x + self.dropout_2(self.multi_head_attention_2(x, encoder_output, encoder_output, first_mask))
    x = self.norm_2(x)

    x = x + self.dropout_3(self.feed_forward(x))
    x = self.norm_3(x)

    return x
