import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from embedding import scaled_dot_product_attention
from Encoder import Encoder
from Decoder import Decoder

class Transformer(nn.Module):
  def __init__(self, src_vocab_size, trc_vocab_size, d_model, N, n_heads):
    super().__init__()
    self.encoder = Encoder(src_vocab_size, d_model, N, n_heads) 
    self.decoder = Decoder(trc_vocab_size, d_model, N, n_heads)
    self.linear = nn.Linear(d_model, trc_vocab_size)

  def forward(self, src, trg, src_mask, trg_mask):
    encoder_output = self.encoder(src, src_mask)
    decoder_output = self.decoder(trg, encoder_output, src_mask, trg_mask)
    output = self.linear(decoder_output)
    return output

    
