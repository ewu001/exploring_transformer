import torch
import torch.nn as nn 
import math

class PositionalEmbedding(nn.Module):
    '''
    Positional embedding injects information about the relative/absolute position of each token in the sequence
    at the bottom of the encoder&decoder network into the input embedding.
    The dimension of positional embedding must be the same as input embedding size so the 2 can be summed.

    There're many ways to construct positional embedding, attention is all you need paper uses the sine and cosine
    functions of different frequencies to approach this.

    Sinusoidal version may allow model to extrapolate to sequences at inference time that are longer than the ones in training
    '''
    def __init__(self, embed_size, max_length):
        '''
        @param embed_size (int)
        @param max_length (int)
        '''
        self.embed_size = embed_size
        self.max_length = max_length
        self.positional_encoding = self.get_positional_encoding()

    def get_positional_encoding(self):
        '''
        Use torch.arange function to initiate tensor of start 0, end max_length,
        as well as tensor of start 0, end embed_size with step of 2
        '''
        position_encoding = torch.zeros(self.max_length, self.embed_size)
        position = torch.arange(0, self.max_length, step=1).unsqueeze(1)
        division_term = 10000**(torch.arange(0, self.embed_size, step=2) / self.embed_size)
        
        position_encoding[:, 0::2] = torch.sin(position / division_term) # pos 0, 2, 4, ... , embed_size
        position_encoding[:, 1::2] = torch.cos(position / division_term) # pos 1, 3, 5, ... , embed_size+1

        return position_encoding.unsqueeze(0)

    def forward(self, inputs):
        '''
        @param input (Tensor): (batch_size, sentence_length, embed_size)
        the output of the embedding projection of input sentence

        returns the concatenated output between input embedding and position embedding
        '''
        length = inputs.size(1)
        # Increase input embedding value to make positional embedding value relatively small
        # To preserve valuable information from input embedding space while still concat positional embedding
        inputs = inputs * math.sqrt(self.embed_size)
        output = inputs + torch.Tensor(self.positional_encoding[:, :length])
        return output