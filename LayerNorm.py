import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from embedding import scaled_dot_product_attention


class LayerNorm(torch.nn.Module):
  def __init__(self, d_model, eps=1e-6):
    super().__init__()
    self.d_model = d_model
    self.alpha = torch.nn.Parameter(torch.ones(self.d_model))
    self.beta = torch.nn.Parameter(torch.zeros(self.d_model))
    self.eps = eps
        
  def forward(self, x):
    x_hat = (x - x.mean(dim=-1, keepdim=True))/(x.std(dim=-1, keepdim=True) + self.eps)
    x_tilde = self.alpha*x_hat + self.beta
    return x_tilde
