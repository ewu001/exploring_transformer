import torch
import torch.nn as nn 

class LayerNormalization(nn.Module):
    '''
    Implements the layer normalization class
    Training deep neural network is computationally expensive and full of unstability, layer norm is similar to 
    batch norm conceptually, to normalize the activities of neuron to achieve stable training effect. 
    The mean and variance used for layer normalization come from all of the summed inputs to the neurons in a layer on a single training case
    '''
    def __init__(self, dim_model, eps=1e-6):

        super(LayerNormalization, self).__init__()
        self.dim_size = dim_model
        self.eps = eps
        # Initialize 2 learnable parameters to calibrate the layer normalization
        # The origin paper of layer norm refers to them as the adaptive gain and bias parameter
        self.gain_param = nn.Parameter(torch.ones(self.dim_size))
        self.bias_param = nn.Parameter(torch.zeros(self.dim_size))

    def forward(self, inputs):
        '''
        returns Tensor of (batch_size, sentence_length, dim_model)
        '''
        mean = torch.mean(inputs, dim=-1, keepdim=True)
        std = torch.std(inputs, dim=-1, keepdim=True)
        normed_input = self.gain_param * (inputs - mean) / (std + self.eps) + self.bias_param
        return normed_input
