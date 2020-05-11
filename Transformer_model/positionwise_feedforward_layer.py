import torch
import torch.nn as nn 

class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_size, d_ff=2048):
        '''
        Based on the paper, embed_size = 512 and d_ff = 2048
        '''
        super(PositionWiseFeedForward, self).__init__()
        self.embed_size = embed_size
        self.d_ff = d_ff
        self.w1_projection = nn.Linear(self.embed_size, self.d_ff, bias=True)
        self.w2_projection = nn.Linear(self.d_ff, self.embed_size, bias=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        '''
        returns Tensor of (batch_size, sentence_length, embed_size)
        '''
        inner_output = torch.nn.functional.relu(self.w1_projection(inputs))
        dropout_output = self.dropout(inner_output)
        output = self.w2_projection(dropout_output)
        return output
