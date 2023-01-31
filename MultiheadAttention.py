import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from embedding import scaled_dot_product_attention



class MultiheadAttention(nn.Module):
  def __init__(self, n_heads, d_model, dropout = 0.1):
    super().__init__()
    self.n_heads = n_heads
    self.d_model = d_model
    self.d_k = d_model // n_heads
    self.d_v = d_model // n_heads

    self.q_linear_layers = []
    self.k_linear_layers = []
    self.v_linear_layers = []

    for i in range(n_heads):
      self.q_linear_layers.append(nn.Linear(in_features = d_model, out_features = self.d_k, bias = False))
      self.k_linear_layers.append(nn.Linear(in_features = d_model, out_features = self.d_k, bias = False))
      self.v_linear_layers.append(nn.Linear(in_features = d_model, out_features = self.d_v, bias = False))

    self.dropout = nn.Dropout(dropout)
    self.unify = nn.Linear(n_heads * self.d_v, d_model)

  def forward(self, q, k, v, mask = None):
    multi_head_attention_outputs = []
    #print((self.q_linear_layers,self.k_linear_layers,self.v_linear_layers))
    for i in range(len(self.q_linear_layers)):
      new_q = self.q_linear_layers[i](q)
      new_k = self.k_linear_layers[i](k)
      new_v = self.v_linear_layers[i](v)
       
      head_v = scaled_dot_product_attention(new_q, new_k, new_v, mask, self.dropout)
      multi_head_attention_outputs.append(head_v)
    
    
    concat = torch.cat(multi_head_attention_outputs, -1)
    output = self.unify(concat)

    return output
