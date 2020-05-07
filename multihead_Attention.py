import torch
import torch.nn as nn 
import math

class MultiheadAttention(nn.Module):
    def __init__(self, dim_model, num_head=8):
        '''
        @param dim_model (int): dimensionality, same as embed_size
        @param num_head (int): number of attention head to concatenate to.
        dimension of the query, key and value will be dim_model / num_head, based on the paper, num_head is 8
        dim_model % num_head must be 0

        '''
        super(MultiheadAttention, self).__init__()
        self.dim_model = dim_model
        self.num_head = num_head
        self.dim_k = self.dim_model / self.num_head
        self.value_projection = nn.Linear(self.dim_model, self.dim_model)
        self.key_projection = nn.Linear(self.dim_model, self.dim_model)
        self.query_projection = nn.Linear(self.dim_model, self.dim_model)
        self.output_projection = nn.Linear(self.dim_model, self.dim_model)


    def forward(self, query, key, value, mask=None):
        '''
        @param query, key, value (Tensor): (batch_size, sentence_length, dim_model)
        Perform linear projection on query, key, value separately
        Split into number of heads
        Send to scaled dot product attention for score
        Concatenate the number of heads together
        return linear projection of the output (batch_size, sentence_length, dim_model)
        '''
        batch_size = query.size(0)
        query_proj = self.query_projection(query)
        key_proj = self.key_projection(key)
        value_proj = self.value_projection(value)

        query_proj = query_proj.view(batch_size, -1, self.num_head, self.dim_k)
        query_proj = query_proj.permute(0, 2, 1, 3)

        key_proj = key_proj.view(batch_size, -1, self.num_head, self.dim_k)
        key_proj = key_proj.permute(0, 2, 1, 3)

        value_proj = value_proj.view(batch_size, -1, self.num_head, self.dim_k)
        value_proj = value_proj.permute(0, 2, 1, 3)

        scores = self.scaledDotProductAttention(query_proj, key_proj, value_proj, mask)
        scores = scores.view(batch_size, -1, self.num_head*self.dim_k)
        output = self.output_projection(scores)
        return output


    def scaledDotProductAttention(self, query, key, value, mask=None):
        '''
        @param query, key, value (Tensor): (batch_size, N_head, sentence_length, dim_k)
        @param mask: (Tensor): (batch_size, sentence_length)
            1 marks position with valid word in the input
            0 marks position with padding
            This is needed before applying the softmax step to reduce the value in our input where it is padded
        Return attention head (Tensor): (batch_size, N_head, sentence_length, dim_k)
        '''
        dim_k = query.size(-1)
        scores = torch.bmm(query, key.permute(0, 1, 3, 2)) / math.sqrt(dim_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, 1e-8)
        # softmax at the sentence_length level
        scores_distribution = torch.nn.functional.softmax(scores, dim=-1)
        attention_head = torch.matmul(scores_distribution, value)
        return attention_head