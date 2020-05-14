import numpy as np
import torch
import torch.nn as nn
import sys

from Transformer_model.encoder import Encoder
from Transformer_model.decoder import Decoder
from Transformer_model.generator import Generator
from Transformer_model.model_embedding import ModelEmbedding
from Transformer_model.positional_Embedding import PositionalEmbedding

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

        self.device = self.get_device

        self.encoder = Encoder(self.dim_model, len(self.vocab.src), self.number_layer)
        self.decoder = Decoder(self.dim_model, len(self.vocab.tgt), self.number_layer)
        self.generator = Generator(self.dim_model, len(self.vocab.tgt))

        # For testing purpose
        self.enc_embed = nn.Embedding(len(self.vocab.src), self.dim_model)
        self.pos_encoder = PositionalEmbedding(self.dim_model, 10000)

        self.dec_embed = nn.Embedding(len(self.vocab.tgt), self.dim_model)
        self.pos_decoder = PositionalEmbedding(self.dim_model, 10000)
        self.torchTransformer = nn.Transformer(d_model=self.dim_model, nhead=self.n_heads, num_encoder_layers=N, num_decoder_layers=N, dim_feedforward=self.dim_model, dropout=0.1, activation='relu')

    def forward(self, src, tgt):
        source_padded = self.vocab.src.to_input_tensor(src, device=self.device)   # Tensor: (sen_len, batch_size )
        #source_padded = source_padded.permute(1, 0)
        src_mask = self.generate_pad_mask(source_padded)

        target_train_padded = self.vocab.tgt.to_input_tensor(tgt, device=self.device)
        #target_padded = target_train_padded.permute(1, 0)  # Tensor: (batch_size, sen_len)
        tgt_pad_mask = self.generate_pad_mask(target_train_padded)
        tgt_square_mask = self.generate_square_subsequent_mask(target_train_padded.size(0))

        src_input = self.enc_embed(source_padded.permute(1, 0))
        src_input = self.pos_encoder(src_input)

        tgt_input = self.dec_embed(target_train_padded.permute(1, 0))
        tgt_input = self.pos_decoder(tgt_input)

        output = self.torchTransformer(src_input.permute(1, 0, 2), tgt_input.permute(1, 0, 2), src_mask=None, tgt_mask=tgt_square_mask,
                                    memory_mask=None, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_pad_mask,
                                    memory_key_padding_mask=src_mask)
        output = self.generator(output)

        # Test print
        #_, words = torch.max(output.permute(1, 0, 2)[-1], dim=-1)
        #print(words.shape)
        #words_to_print = [self.vocab.tgt.id2word[i] for i in words.squeeze().tolist()]
        #print("Sample predicted sentence: ", words_to_print)

        return output

    """
    def forward(self, src, tgt):
        '''
        Take a mini-batch of source and target sentences, compute the log-likelihood of target sentences under the
        language model learned by the transformer network
        @returns scores (Tensor): a variable/tensor of shape (batch_size, ) representing the log-likelihood of generating
        the gold-standard target sentence for each example in the input batch.

        '''
        # Convert list of lists into tensors
        #tgt_to_train = [sent[:-1] for sent in tgt]
        #tgt_to_validate = [sent[1:] for sent in tgt]

        source_padded = self.vocab.src.to_input_tensor(src, device=self.device)   # Tensor: (sen_len, batch_size )
        source_padded = source_padded.permute(1, 0)
        src_mask = generate_src_masks(source_padded, self.vocab.src['<pad>'])
        #print(src_mask)
        encoder_output = self.encoder(source_padded, src_mask) # Tensor: (batch_size, src_len, dim_model)
        #print("encoder output shape: ", encoder_output.shape)
        target_train_padded = self.vocab.tgt.to_input_tensor(tgt, device=self.device)
        target_padded = target_train_padded.permute(1, 0)  # Tensor: (batch_size, sen_len)
        #print("target_padded shape: ", target_padded.shape)

        # From here:
        # Loop through based on sen_len, it should be the longest sentence in current batch, each iteration serves as one time step
        # Cut target_padded based on loop index so each time step the decoder sees one more target word
        # Compute target mask based on incremental length of decoder input
        # get decoder output for this time step
        # add this current max output to a combined output list, note this list is apart from the ground truth target list
        # zero out, probabilities for which we have nothing in the target text, such as pad token
        # compute log probability of generating true target words
        #target_seq_length = target_padded.size(1)
        #print("target_seq_length: ", target_seq_length)
        #print(" Current input target is: ", tgt)
        '''
        combined_outputs = []
        print_output = []
        print(" Current input target is: ", tgt)
        for i in range(target_seq_length):
            current_decoder_input = target_padded[:, :i+1]
            #print("current decoder_input shape: ", current_decoder_input.shape)
            current_target_mask = generate_tgt_masks(current_decoder_input, self.vocab.tgt['<pad>'])
            current_decoder_output = self.decoder(current_decoder_input, encoder_output, src_mask, current_target_mask)
            #print("current decoder output shape: ", current_decoder_output.shape)
            current_output = current_decoder_output[:, -1, :]
            #print("insert decoder output shape: ", current_output.shape)

            combined_outputs.append(current_output)
            #current_output = self.generator(current_decoder_output)  # Tensor (batch size, 1, tgt_vocab_size)
            #target_padded[:, i+1] = current_decoder_output

            output_dist = torch.nn.functional.log_softmax(current_output, dim=-1)
            _, next_word = torch.max(output_dist, dim=-1)
            # print(next_word)
            next_word_id = next_word.data[0].item()
            print_output.append(self.vocab.tgt.id2word[next_word_id])

        combined_outputs = torch.stack(combined_outputs) # Tensor ( tgt_length, batch_size, d_model )
        #print("combined outputs shape: ", combined_outputs.shape)
        prob_dist = self.generator(combined_outputs)
        #print("prob dist shape: ", prob_dist.shape) # Tensor ( tgt_length, batch_size, tgt_vocab_size)
        print(" Current predicted target is: ", print_output)
        '''
        target_mask = generate_tgt_masks(target_padded, self.vocab.tgt['<pad>'])
        #print(target_mask)
        #print("target_mask in training: ", target_mask.shape)
        decoder_output = self.decoder(target_padded, encoder_output, src_mask, target_mask)
        # print("decoder_output shape: ", decoder_output.shape)  # (batch_size, tgt_length, d_model)
        final_output = self.generator(decoder_output)  # (batch_size, tgt_length, tgt_vocab_size)

        # Test print
        _, words = torch.max(final_output[-1], dim=-1)
        print(words.shape)
        words_to_print = [self.vocab.tgt.id2word[i] for i in words.squeeze().tolist()]
        print("Sample predicted sentence: ", words_to_print)

        # Zero out, probabilities for which we have nothing in the target text
        final_output = final_output.permute(1, 0, 2) # (sentence_len, batch_size, vocab_size)

        #target_to_val_padded = self.vocab.tgt.to_input_tensor(tgt_to_validate, device=self.device) # (sen_len, batch_size )

        #target_to_val_pad_masks = (target_to_val_padded != self.vocab.tgt['<pad>']).float()
        # Compute log probability of generating true target words
        #target_gold_words_log_prob = torch.gather(final_output, index=target_to_val_padded.unsqueeze(-1), dim=-1).squeeze(-1) * target_to_val_pad_masks
        #scores = target_gold_words_log_prob.sum(dim=0)

        return final_output
    """

    def generate_pad_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

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
        #print("encoder output: ", enc_outputs.shape)

        target_output = []
        target_output.append('<s>')
        for i in range(1, max_decoding_time_step):
            
            tgt_mask = np.triu(np.ones((1, i, i)),k=1).astype('uint8')
            if torch.cuda.is_available():
                tgt_mask = torch.autograd.Variable(torch.from_numpy(tgt_mask) == 0).cuda()  # For GPU
            else:
                tgt_mask = torch.autograd.Variable(torch.from_numpy(tgt_mask) == 0)  # For CPU

            current_output = target_output

            tgt_word_ids = self.vocab.tgt.words2indices(current_output)
            target_tensor = torch.tensor(tgt_word_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            print("target tensor shape: ", target_tensor.shape)
            
            #print(target_tensor.shape)  # Tensor: (batch_size, sent_length )
            decoder_output = self.decoder(target_tensor, enc_outputs, None, tgt_mask)
            output_dist = self.generator(decoder_output)
            prediction = output_dist[:, -1, :]  # (batch_size, 1, vocab_size)
            #output_dist = torch.nn.functional.log_softmax(prediction, dim=-1)
            #print("decoder out: ", output_dist.shape)

            _, next_word = torch.max(prediction, dim=-1)
            print("next word: ", next_word)
            next_word_id = next_word.data[0].item()
            target_output.append(self.vocab.tgt.id2word[next_word_id])
            if next_word_id == self.vocab.tgt.word2id['</s>']:
                break
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
            print(hypotheses)
            exp_enc_outputs = enc_outputs.expand(hyp_num,
                                                     enc_outputs.size(1),
                                                     enc_outputs.size(2))

            tgt_mask = np.triu(np.ones((1, t, t)), k=1).astype('uint8')
            if torch.cuda.is_available():
                tgt_mask = torch.autograd.Variable(torch.from_numpy(tgt_mask) == 0).cuda()  # For GPU
            else:
                tgt_mask = torch.autograd.Variable(torch.from_numpy(tgt_mask) == 0)  # For CPU

            #current_output = target_output[:t]
            tgt_word_ids = [self.vocab.tgt.words2indices(hyp) for hyp in hypotheses]
            target_tensor = torch.tensor(tgt_word_ids, dtype=torch.long, device=self.device)



            print(target_tensor.shape)  # Tensor: (batch_size, sent_length )
            print(exp_enc_outputs.shape)  # Tensor: (batch_size, sent_length )
            decoder_output = self.decoder(target_tensor, exp_enc_outputs, None, tgt_mask)
            output = self.generator(decoder_output)
            prediction = output[:, -1, :]
            print(prediction.shape)

            # log probabilities over target words
            #log_p_t = torch.nn.functional.log_softmax(output, dim=-1)
            #log_p_t = prediction.permute(1, 0, 2)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(prediction) + prediction).view(-1)
            print(contiuating_hyp_scores.shape)
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
