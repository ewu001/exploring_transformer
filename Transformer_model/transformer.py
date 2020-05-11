import numpy as np
import torch
import torch.nn as nn
import sys

from Transformer_model.encoder import Encoder
from Transformer_model.decoder import Decoder
from Transformer_model.generator import Generator
from Transformer_model.model_embedding import ModelEmbedding

from utility import generate_src_masks, generate_tgt_masks
from collections import namedtuple
from typing import List, Tuple
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class Transformer(nn.Module):
    '''
    Provides abstraction for encoder to decoder architecture,
    Encoder takes input (batch, sentence_length), projects to embedding space then outputs 
    a continuous representation (batch, sentence_length, embed_size)
    Decoder takes encoder input, generates output at each step via auto-regressive pattern
    Generator includes a linear projection of decoder output and map to softmax regression for output probability distribution
    '''
    def __init__(self, vocab, dim_model, n_heads, N=1):
        super(Transformer, self).__init__()
        self.vocab = vocab
        self.dim_model = dim_model
        self.n_heads = n_heads
        self.number_layer = N
        self.model_embedding = ModelEmbedding(self.dim_model, len(self.vocab.src))

        self.device = self.get_device

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
            'args': dict(dim_model=self.dim_model, n_heads=self.n_heads,  N=self.number_layer),
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

    @property
    def get_device(self):
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Additional Info when using cuda
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
        return device


    def greedy_decoding(self, src_sent, max_decoding_time_step):
        '''
        Given a source sentence, perform greedy decode algorithm to yield output sequence in the target language
        @param src_sent: lis[str], one sentence to decide
        '''
        #src_mask = (src_sent != input_pad).unsqueeze(-2)
        word_ids = self.vocab.src.words2indices(src_sent)

        source_tensor = torch.tensor(word_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # Tensor: (batch_size, sent_length )

        enc_outputs = self.encoder(source_tensor, None)
        print("encoder output: ", enc_outputs.shape)

        #outputs = torch.zeros(max_decoding_time_step)
        target_output = []
        target_output.append('<s>')
        for i in range(1, max_decoding_time_step):    
            
            #tgt_mask = np.triu(np.ones((1, i, i)),k=1).astype('uint8')
            #if torch.cuda.is_available():
            #    tgt_mask = torch.autograd.Variable(torch.from_numpy(tgt_mask) == 0).cuda()  # For GPU
            #else:
            #    tgt_mask = torch.autograd.Variable(torch.from_numpy(tgt_mask) == 0)  # For CPU

            current_output = target_output

            tgt_word_ids = self.vocab.tgt.words2indices(current_output)
            target_tensor = torch.tensor(tgt_word_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            
            #print(target_tensor.shape)  # Tensor: (batch_size, sent_length )
            decoder_output = self.decoder(target_tensor, enc_outputs, None, None)
            output_dist = self.generator(decoder_output)
            #output_dist = torch.nn.functional.log_softmax(output, dim=-1)
            #print("decoder out: ", output_dist.shape)

            _, next_word = torch.max(output_dist, dim=-1)
            next_word_id = next_word.data[0][0].item()
            target_output.append(self.vocab.tgt.id2word[next_word_id])
            #top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(output_dist, k=1)
            #print(top_cand_hyp_scores)
            #print(top_cand_hyp_pos)
            #target_output.append(top_cand_hyp_pos[0][0].item())
            #if top_cand_hyp_pos[0][0] == self.vocab.tgt.word2id['</s>']:
            #    break
        print(target_output)
        #target_output_extraction = [self.vocab.tgt.id2word[ix] for ix in target_output]
        # Due to greedy decoding, only 1 hypothesis will be generated thus the score is equal to 1
        # Score value will vary in the case of beam search decoding
        hypothesis = Hypothesis(value=target_output, score=1)
        
        return [hypothesis]


    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        word_ids = self.vocab.src.words2indices(src_sent)
        source_tensor = torch.tensor(word_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # Tensor: (batch_size, sent_length )

        enc_outputs = self.encoder(source_tensor, None)

        target_output = [0 for i in range(max_decoding_time_step)]
        target_output[0] = self.vocab.tgt.word2id['<s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_enc_outputs = enc_outputs.expand(hyp_num,
                                                     enc_outputs.size(1),
                                                     enc_outputs.size(2))

            tgt_mask = np.triu(np.ones((1, t, t)), k=1).astype('uint8')
            if torch.cuda.is_available():
                tgt_mask = torch.autograd.Variable(torch.from_numpy(tgt_mask) == 0).cuda()  # For GPU
            else:
                tgt_mask = torch.autograd.Variable(torch.from_numpy(tgt_mask) == 0)  # For CPU

            #current_output = target_output[:t]
            tgt_word_ids = self.vocab.src.words2indices(hypotheses)
            target_tensor = torch.tensor(tgt_word_ids, dtype=torch.long, device=self.device).unsqueeze(0)

            # print(target_tensor.shape)  # Tensor: (batch_size, sent_length )
            decoder_output = self.decoder(target_tensor, exp_enc_outputs, None, tgt_mask)
            output = self.generator(decoder_output)



            # log probabilities over target words
            log_p_t = torch.nn.functional.log_softmax(output, dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses
