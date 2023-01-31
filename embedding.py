import torch
import torch.nn as nn
import math
import torch.nn.functional as F



class Embedder(nn.Module):
  def __init__(self, vocab_size, d_model):
    super().__init__()  
    self.embed = nn.Embedding(vocab_size, d_model)
  
  def forward(self, x):
    return self.embed(x)

class PositionalEncoder(torch.nn.Module):
  def __init__(self, d_model, max_seq_len=80):
    super().__init__()
    self.d_model = d_model
    pe_matrix = torch.zeros(max_seq_len, d_model)
    for pos in range(max_seq_len):
      for i in range(0, d_model, 2):
        pe_matrix[pos, i] = math.sin(pos/10000**(2*i/d_model))
        pe_matrix[pos, i+1] = math.cos(pos/10000**(2*i/d_model))
    pe_matrix = pe_matrix.unsqueeze(0) 
    self.register_buffer('pe', pe_matrix)
        
  def forward(self, x):
    seq_len = x.size()[1]
    x = x + self.pe[:, :seq_len]
    return x

def scaled_dot_product_attention(q, k, v, mask = None, dropout = None):
  att_scores =  torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])

  #if mask is not None:
  #  att_scores = att_scores.masked_fill(mask == 0, value = -1e9)
  

  att_weights = F.softmax(att_scores, dim = -1)

  if dropout is not None:
    att_weights = dropout(att_weights)

  output = torch.matmul(att_weights, v)
  return output
