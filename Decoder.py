import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from embedding import Embedder, PositionalEncoder
from Clone import clone_layer
from DecoderLayer import DecoderLayer
from LayerNorm import LayerNorm

class Decoder(nn.Module):
  def __init__(self, vocab_size, d_model, N, n_heads):
    super().__init__()
    self.embed = Embedder(vocab_size, d_model)
    self.pe = PositionalEncoder(d_model)
    self.decoder_layers = clone_layer(DecoderLayer(d_model, n_heads), N)
    self.norm = LayerNorm(d_model)

  def forward(self, trg, encoder_output, src_mask, trg_mask):
    x = self.embed(trg)
    x = self.pe(x)
    for decoder in self.decoder_layers:
      x = decoder(x, encoder_output, src_mask, trg_mask)
    return self.norm(x)
    
