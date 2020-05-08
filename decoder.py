import torch 
import torch.nn as nn 
import utility

from model_embedding import ModelEmbedding
from positional_Embedding import PositionalEmbedding
from positionwise_feedforward_layer import PositionWiseFeedForward
from multihead_Attention import MultiheadAttention
from layer_norm import LayerNormalization

class Decoder(nn.Module):
    '''
    Implementation of the decoder network from transformer, there are 6 total decoder layers stack on top of each other
    '''
    def __init__(self, dim_model, tgt_vocab_size, num_decoder=6):
        '''
        @param dim_model (int): dimensionality of the input embedding space, according to paper it should be 512
        @param tgt_vocab_size (int): dimensionality of the total size of the target vocabulary size
        @param num_decoder (int): number of total decoder layers stacking on top of each other
        '''
        super(Decoder, self).__init__()
        self.dim_model = dim_model
        self.tgt_vocab_size = tgt_vocab_size
        self.num_decoder = num_decoder
        self.input_embedding = ModelEmbedding(self.dim_model, self.tgt_vocab_size)
        self.position_embedding = PositionWiseFeedForward(self.dim_model)

        self.decoder_layer = DecoderLayer(self.dim_model, num_head=8)
        self.layerNorm = LayerNormalization(self.dim_model)

        self.layers = utility.get_clones(self.decoder_layer, self.num_decoder)

    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        '''
        This forward will take care of the entire transformer decoder's computation
        Input sequence first go through embedding projection, followed by positional embedding concatenation
        Then go into each of the encoder layer based on number of encoder
        '''
        embed_input = self.input_embedding(tgt)
        x = self.position_embedding(embed_input)
        for dec_layer in self.layers:
            x = dec_layer(x, enc_output, tgt_mask, src_mask)
            x = self.layerNorm(x)
        return x


class DecoderLayer(nn.Module):
    '''
    Implementation of one single transformer decoder component
    Input: with positional encoding added
    One masked multi-head attention computes self attention with input as query, key, value at the same time
    One residual network with input plus masked self attention output
    One multi-head attention with query and key coming from encoder output, and value come from previous masked attention layer
    One residual network with layer normalization
    One feed forward network
    One residual network with layer normalization
    Returns decoder output for generator to compute probability distribution over target category
    '''
    def __init__(self, dim_model, num_head):
        super(DecoderLayer, self).__init__()
        self.dim_model = dim_model
        self.num_head = num_head

        self.feedForward = PositionWiseFeedForward(self.dim_size)
        self.maskedMultiheadAttention = MultiheadAttention(self.dim_model, self.num_head)
        self.multiheadAttention = MultiheadAttention(self.dim_model, self.num_head)

        self.layerNorm_1 = LayerNormalization(self.dim_model)
        self.layerNorm_2 = LayerNormalization(self.dim_model)  
        self.layerNorm_3 = LayerNormalization(self.dim_model)  

    def forward(self, x, encoder_output, tgt_mask, src_mask):
        '''
        Here the input x is already added with positional encoding
        Decoder layer forward should first distribute input x into query, key and value for multi-head attention to compute self attention, with provided mask
        Then concatenate with the original input via a residual network, followed by a layer normalization
        Then go through a feed forward network with 2 layers
        Then concatenate with its input via a second residual network, before returning it as output
        '''
        masked_attention = self.maskedMultiheadAttention(x, x, x, tgt_mask)
        first_add_norm = self.layerNorm_1(masked_attention + x)
        # Query comes from previous decoder layer while the key and value comes from encoder output
        # All position in decoder can attend over all position of input sequence
        # Here we use source mask to compute scaled dot product attention heads
        self_attention = self.multiheadAttention(first_add_norm, encoder_output, encoder_output, src_mask)
        second_add_norm = self.layerNorm_2(first_add_norm + self_attention)
        feed_forward = self.feedForward(second_add_norm)
        third_add_norm = self.layerNorm_3(second_add_norm + feed_forward)
        return third_add_norm