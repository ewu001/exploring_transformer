import numpy as np
import torch
import torch.nn as nn
import sys

from Transformer_model.encoder import Encoder
from Transformer_model.decoder import Decoder
from Transformer_model.generator import Generator

from utility import generate_src_masks, generate_tgt_masks
from collections import namedtuple
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class Transformer(nn.Module):
    '''
    Provides abstraction for encoder to decoder architecture,
    Encoder takes input (batch, sentence_length), projects to embedding space then outputs 
    a continuous representation (batch, sentence_length, embed_size)
    Decoder takes encoder input, generates output at each step via auto-regressive pattern
    Generator includes a linear projection of decoder output and map to softmax regression for output probability distribution
    '''
    def __init__(self, vocab, dim_model, n_heads, device, N=6):
        super(Transformer, self).__init__()
        self.vocab = vocab
        self.dim_model = dim_model
        self.n_heads = n_heads
        self.number_layer = N
        # Hardcode device to CPU for now
        self.device = device

        self.encoder = Encoder(self.dim_model, len(self.vocab.src), self.number_layer)
        self.decoder = Decoder(self.dim_model, len(self.vocab.tgt), self.number_layer)
        self.generator = Generator(self.dim_model, len(self.vocab.tgt))

    def forward(self, src, tgt):

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(src, device=self.device)   # Tensor: (sen_len, batch_size )
        source_padded = source_padded.permute(1, 0)
        src_mask = generate_src_masks(source_padded, self.vocab.src['<pad>'])
        target_padded = self.vocab.tgt.to_input_tensor(tgt, device=self.device)   # Tensor: (sen_len, batch_size)
        target_padded = target_padded.permute(1, 0)
        tgt_mask = generate_tgt_masks(target_padded, self.vocab.tgt['<pad>'])
        encoder_output = self.encoder(source_padded, src_mask) # Tensor: (batch_size, src_len, dim_model)
        decoder_output = self.decoder(target_padded, encoder_output, src_mask, tgt_mask)
        output = self.generator(decoder_output)  # Tensor (batch size, tgt_length, tgt_vocab_size)

        # Zero out, probabilities for which we have nothing in the target text
        #target_masks = (target_padded != self.vocab.tgt['<pad>']).float()
        
        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(output, index=target_padded.unsqueeze(-1), dim=-1)
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores


    def save(self, path: str):
        """ Save the trained model to a file at specified path location.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(dim_model=self.dim_model, n_heads=self.n_heads, device=self.device, N=self.number_layer),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


    @staticmethod
    def load(model_path: str):
        """ Load the model file from specified path.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = Transformer(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model


    def greedy_decoding(self, src_sent, max_decoding_time_step):
        '''
        Greedy decode implementation
        @param src_sent: lis[str], one sentence to decide
        '''
        #src_mask = (src_sent != input_pad).unsqueeze(-2)
        word_ids = self.vocab.src.words2indices(src_sent)   
        source_tensor = torch.tensor(word_ids, dtype=torch.long, device=self.device).unsqueeze(0) # Tensor: (batch_size, sent_length )

        #print(source_tensor.shape)
        enc_outputs = self.encoder(source_tensor, None)
    
        #outputs = torch.zeros(max_decoding_time_step)
        target_output = [0 for i in range(max_decoding_time_step)]
        target_output[0] = self.vocab.tgt.word2id['<s>']
        for i in range(1, max_decoding_time_step):    
            
            tgt_mask = np.triu(np.ones((1, i, i)),k=1).astype('uint8')
            #trg_mask= torch.autograd.Variable(torch.from_numpy(trg_mask) == 0).cuda()  # For GPU
            tgt_mask = torch.autograd.Variable(torch.from_numpy(tgt_mask) == 0)  # For CPU
            current_output = target_output[:i]
            tgt_word_ids = self.vocab.src.words2indices(current_output)   
            target_tensor = torch.tensor(tgt_word_ids, dtype=torch.long, device=self.device).unsqueeze(0) 
            
            #print(target_tensor.shape)  # Tensor: (batch_size, sent_length )
            decoder_output = self.decoder(target_tensor, enc_outputs, None, tgt_mask)
            output = self.generator(decoder_output)
            out = torch.nn.functional.softmax(output, dim=-1)
            _, ix = out[:, -1].data.topk(1)

            target_output[i] = ix[0][0].item()
            if ix[0][0] == self.vocab.tgt.word2id['</s>']:
                break

        return [self.vocab.tgt.id2word[ix] for ix in target_output if target_output[ix] not in [1, 0]]

