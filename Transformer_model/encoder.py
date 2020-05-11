import torch 
import torch.nn as nn 
import utility

from Transformer_model.model_embedding import ModelEmbedding
from Transformer_model.positional_Embedding import PositionalEmbedding
from Transformer_model.positionwise_feedforward_layer import PositionWiseFeedForward
from Transformer_model.multihead_Attention import MultiheadAttention
from Transformer_model.layer_norm import LayerNormalization

class Encoder(nn.Module):
    '''
    Implementation of the encoder network from transformer, there are 6 total encoder layers stack on top of each other
    '''
    def __init__(self, dim_model, src_vocab_size, num_encoder=6):
        '''
        @param dim_model (int): dimensionality of the input embedding space, according to paper it should be 512
        @param vocab_size (int): dimensionality of the total size of source vocabulary size
        @param num_encoder (int): number of total encoder layers stacking on top of each other
        '''
        super(Encoder, self).__init__()
        self.dim_model = dim_model
        self.src_vocab_size = src_vocab_size
        self.num_encoder = num_encoder
        self.input_embedding = ModelEmbedding(self.dim_model, self.src_vocab_size)
        self.position_embedding = PositionalEmbedding(self.dim_model, max_length=10000)
        self.encoderLayer = EncoderLayer(self.dim_model, num_head=8)
        self.layerNorm = LayerNormalization(self.dim_model)
        self.layers = utility.get_clones(self.encoderLayer, self.num_encoder)


    def forward(self, src, mask):
        '''
        This forward will take care of the entire transformer encoder's computation
        Input sequence first go through embedding projection, followed by positional embedding concatenation
        Then go into each of the encoder layer based on number of encoder
        @param src (Tensor): (batch_size, sentence_length)
        '''
        embed_input = self.input_embedding(src)
        x = self.position_embedding(embed_input)
        for enc_layer in self.layers:
            x = enc_layer(x, mask)
            x = self.layerNorm(x)
        return x


class EncoderLayer(nn.Module):
    '''
    Implementation of one single transformer encoder component
    Input: with positional encoding added
    One multi-head attention computes self attention with input as query, key, value at the same time
    One residual network with input plus self attention output
    One feed forward layer 
    One residual network
    Returns encoder output
    '''
    def __init__(self, dim_model, num_head):
        super(EncoderLayer, self).__init__()
        self.dim_model = dim_model
        self.num_head = num_head
        self.multiheadAttention = MultiheadAttention(self.dim_model, self.num_head)
        self.feedForward = PositionWiseFeedForward(self.dim_model)

        self.layerNorm_1 = LayerNormalization(self.dim_model)
        self.layerNorm_2 = LayerNormalization(self.dim_model)
        self.dropout = nn.Dropout(0.1)


    def forward(self, x, mask):
        '''
        Encoder layer forward should first distribute input x into query, key and value for multi-head attention to compute self attention, with provided mask
        Then concatenate with the original input via a residual network, followed by a layer normalization
        Then go through a feed forward network with 2 layers
        Then concatenate with its input via a second residual network, before returning it as output
        '''
        self_attention = self.multiheadAttention(x, x, x, mask)
        first_add_norm = self.layerNorm_1(self_attention + x)
        second_add_norm = self.layerNorm_2(first_add_norm + self.dropout(self.feedForward(first_add_norm)))
        return second_add_norm