import numpy as np
import torch
import torch.nn as nn

from Transformer_model.encoder import Encoder
from Transformer_model.decoder import Decoder
from Transformer_model.generator import Generator

from utility import generate_src_masks, generate_tgt_masks

class Transformer(nn.Module):
    '''
    Provides abstraction for encoder to decoder architecture,
    Encoder takes input (batch, sentence_length), projects to embedding space then outputs 
    a continuous representation (batch, sentence_length, embed_size)
    Decoder takes encoder input, generates output at each step via auto-regressive pattern
    Generator includes a linear projection of decoder output and map to softmax regression for output probability distribution
    '''
    def __init__(self, vocab, dim_model, n_heads, N=6):
        super(Transformer, self).__init__()
        self.vocab = vocab
        self.dim_model = dim_model
        self.n_heads = n_heads
        self.number_layer = N

        self.encoder = Encoder(self.dim_model, len(self.vocab.src), self.number_layer)
        self.decoder = Decoder(self.dim_model, len(self.vocab.tgt), self.number_layer)
        self.generator = Generator(self.dim_model, len(self.vocab.tgt))

    def forward(self, src, tgt):
        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(src, device=self.device)   # Tensor: (batch_size, sen_len)
        src_mask = generate_src_masks(source_padded, self.vocab.src['<pad>'])
        
        target_padded = self.vocab.tgt.to_input_tensor(tgt, device=self.device)   # Tensor: (batch_size, sen_len)
        tgt_mask = generate_tgt_masks(target_padded, self.vocab.tgt['<pad>'])

        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.generator(decoder_output)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()
        
        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(output, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores


