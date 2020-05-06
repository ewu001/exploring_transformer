import torch
import torch.nn as nn 
import math

class ModelEmbedding(nn.Module):
    '''
    This class projects the input into the embedding dimension
    '''
    def __init__(self, embed_size, vocab_size):
        '''
        Init the Embedding layers.
        @param embed_size (int): Embedding size (dimensionality)
        @param vocab_size (int): Vocabulary size (dimensionality)

        For transformer, the same learned embedding weight matrix is shared between encoder and decoder embedding layer
        And based on the paper's instruction,
        inside the embedding layer, the weights are multiplied by square root of embed size
        '''
        super(ModelEmbedding, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

    def forward(self, input):
        '''
        @param input (Tensor):  often of shape (batch_size, sentence_length)
        Returns output: (Tensor): (batch_size, sentence_length, embed_size)
        '''
        input_embed = self.embedding(input)
        output = input_embed * math.sqrt(self.embed_size)
        return output