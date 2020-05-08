import numpy as np
import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from generator import Generator

class Transformer(nn.Module):
    '''
    Provides abstraction for encoder to decoder architecture,
    Encoder takes input (batch, sentence_length), projects to embedding space then outputs 
    a continuous representation (batch, sentence_length, embed_size)
    Decoder takes encoder input, generates output at each step via auto-regressive pattern
    Generator includes a linear projection of decoder output and map to softmax regression for output probability distribution
    '''
    def __init__(self, src_vocab, tgt_vocab, dim_model, n_heads, N=6):
        super(Transformer, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.dim_model = dim_model
        self.n_heads = n_heads
        self.number_layer = N

        self.encoder = Encoder(self.dim_model, self.src_vocab, self.num_layer)
        self.decoder = Decoder(self.dim_model, self.tgt_vocab, self.num_layer)
        self.generator = Generator(self.dim_model, self.tgt_vocab)

    def forward(self, src, src_mask, tgt, tgt_mask):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.generator(decoder_output)
        return output


