import torch
import torch.nn as nn

class Generator(nn.Module):
    '''
    This class projects the output of the decoder onto the probability distribution over the entire space of vocab size
    '''
    def __init__(self, dim_model, vocab_size):
        '''
        @param dim_model (int): dimension of the model, often refered to as the embed size across this implementation
        @param vocab_size (int): dimension of the total vocabulary space
        '''
        super(Generator, self).__init__()
        self.dim_model = dim_model
        self.vocab_size = vocab_size
        self.target_vocab_projection = torch.nn.Linear(self.dim_model, self.vocab_size)

    def forward(self, inputs):
        '''
        Linear projection of decoder output, followed by log softmax to output probability distribution
        @param inputs (Tensor): (batch_size, sentence_length, hidden_size)

        
        '''
        linear_output = self.target_vocab_projection(inputs)
        #softmax_output = torch.nn.functional.log_softmax(linear_output, dim=-1)


        return linear_output