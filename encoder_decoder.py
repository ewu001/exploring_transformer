import numpy as np
import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    '''
    Provides abstraction for encoder to decoder architecture,
    Encoder takes input (batch, sentence_length), projects to embedding space then outputs 
    a continuous representation (batch, sentence_length, embed_size)
    Decoder takes encoder input, generates output at each step via auto-regressive pattern
    Generator includes a linear projection of decoder output and map to softmax regression for output probability distribution
    '''
    def __init__(self, encoder, decoder, generator, src_embed, tgt_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def forward(self, src, src_mask, tgt, tgt_mask):
        '''
        
        '''
        pass

    def encode(self, src, src_mask):
        pass

    def decode(self, tgt, tgt_mask):
        pass

