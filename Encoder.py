import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from embedding import Embedder, PositionalEncoder
from Clone import clone_layer
from DecoderLayer import DecoderLayer
from LayerNorm import LayerNorm
from EncoderLayer import EncoderLayer


class Encoder(nn.Module):
  def __init__(self, vocab_size, d_model, N, n_heads):
    super().__init__()
    self.embed = Embedder(vocab_size, d_model)
    self.pe = PositionalEncoder(d_model)
    self.encoder_layers = clone_layer(EncoderLayer(d_model, n_heads), N)
    self.norm = LayerNorm(d_model)

  def forward(self, src, mask):
    x = self.embed(src)
    x = self.pe(x)
    for encoder in self.encoder_layers:
      x = encoder(x, mask)

    return self.norm(x)
