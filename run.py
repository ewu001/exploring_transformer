#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Borrowed from Stanford NLP course
Usage:
    run.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
    --pretrain-model=<file>                 if provided, load pretrain model instead of start fresh
"""
import math
import sys
import pickle
import time
import argparse


from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from collections import namedtuple
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utility import read_corpus, batch_iter
from vocab import Vocab, VocabEntry
from Transformer_model.transformer import Transformer
from Transformer_model.torchTransformer import TorchTransformer

import torch
import torch.nn as nn
import torch.nn.utils

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

def evaluate_ppl(model, dev_data, vocab, device, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (Transformer): Transformer Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            
            tgt_input = [ i[:-1] for i in tgt_sents]
            tgt_real = [ i[1:] for i in tgt_sents ]


            tgt_real_value = vocab.tgt.to_input_tensor(tgt_real, device=device)

            #loss = -model(src_sents, tgt_sents).sum()
            prediction = model(src_sents, tgt_input) # (tgt_sentence, batch_size, tgt_vocab_size)

            # Test print
            _, words = torch.max(prediction.permute(1, 0, 2)[-1], dim=-1)
            words_to_print = [vocab.tgt.id2word[i] for i in words.squeeze().tolist()]
            print("Sample predicted sentence: ", words_to_print)

            vocab_size = prediction.shape[-1]

            # Generate LOSS
            log_pred = torch.nn.functional.log_softmax(prediction, dim=-1)
            tgt_real_value = vocab.tgt.to_input_tensor(tgt_real, device=device)
            target_to_val_pad_masks = (tgt_real_value != vocab.tgt.word2id['<pad>']).float()
            # Compute log probability of generating true target words
            target_gold_words_log_prob = torch.gather(log_pred, index=tgt_real_value.unsqueeze(-1), dim=-1).squeeze(-1) * target_to_val_pad_masks
            batch_loss_scores = target_gold_words_log_prob.sum(dim=0)

            cum_loss += (-batch_loss_scores).sum()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = math.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])
    return bleu_score


def train(args: Dict):
    """ Train the Transformer Model.
    @param args (Dict): args from cmd line
    """

    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    #train_batch_size = int(args['--batch-size'])
    #valid_niter = int(args['--valid-niter'])
    train_batch_size = 2
    valid_niter = 2000
    clip_grad = float(args['--clip-grad'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    maxEpoch = int(args['--max-epoch'])
    vocab = Vocab.load(args['--vocab'])

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    # Init the transformer instance
    #model = Transformer(vocab=vocab, dim_model=256, n_heads=8)   # Native transformer class
    model = TorchTransformer(vocab=vocab, dim_model=256, n_heads=8)  # uses torch transformer api

    model.train()

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.tgt.word2id['<pad>'])

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    if args['--pretrain-model']:
        # load model from previous training
        pretrain_model = args['--pretrain-model']
        params = torch.load(pretrain_model, map_location=lambda storage, loc: storage)
        model.load_state_dict(params['state_dict'])
        model = model.to(device)
        print('restore from previous trained model', file=sys.stderr)

        print('restore parameters of the optimizers', file=sys.stderr)
        optimizer.load_state_dict(torch.load(pretrain_model + '.optim'))

        # decay lr, and restore from previously best checkpoint
        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

        model_save_path = pretrain_model
        # set new lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_sents)

            tgt_input = [ i[:-1] for i in tgt_sents]
            tgt_real = [ i[1:] for i in tgt_sents ]

            prediction = model(src_sents, tgt_input) # (tgt_sentence, batch_size, tgt_vocab_size)

            # Generate LOSS
            log_pred = torch.nn.functional.log_softmax(prediction, dim=-1)
            tgt_real_value = vocab.tgt.to_input_tensor(tgt_real, device=device)
            target_to_val_pad_masks = (tgt_real_value != vocab.tgt.word2id['<pad>']).float()
            # Compute log probability of generating true target words
            target_gold_words_log_prob = torch.gather(log_pred, index=tgt_real_value.unsqueeze(-1), dim=-1).squeeze(-1) * target_to_val_pad_masks
            batch_loss_scores = target_gold_words_log_prob.sum(dim=0)

            batch_loss = (-batch_loss_scores).sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, dev_data, vocab, device, batch_size=4)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == maxEpoch:
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def decode(args: Dict[str, str]):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """

    print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        print("load test target sentences from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    model = Transformer.load(args['MODEL_PATH'])

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print('Corpus BLEU: {}'.format(bleu_score * 100), file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def beam_search(model: Transformer, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (Transformer): Transformer Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
            #example_hyps = model.greedy_decoding(src_sent, max_decoding_time_step=max_decoding_time_step)
            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses


def main():
    """ Main func.
    """

    args = docopt(__doc__)

    # Check pytorch version
    assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # seed the random number generators
    #seed = int(args['--seed'])
    seed = 42
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
