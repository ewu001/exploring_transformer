import torch
import torch.nn as nn
import copy
import nltk
import math
import numpy as np
#nltk.download('punkt')

# Build a convenient cloning function that can generate multiple encoder or decoder layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    
    returns list of tokenized sentences (list[list[str]])
    """
    data = []
    for line in open(file_path):
        sent = nltk.word_tokenize(line)
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data

def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of each sentence.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    # Find max_length from input batch sentence
    len_x = [len(x) for x in sents]
    len_x.sort()
    max_length = len_x[-1]
    for sentence in sents:
        new_sentence = sentence + [pad_token] * (max_length - len(sentence))
        sents_padded.append(new_sentence)

    return sents_padded

def batch_iter(data, batch_size, shuffle=False):
    """ Turns batches of source and target sentences reverse sorted by length of descending order (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """

    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

def generate_src_masks(source_padded, pad):
    """ Generate sentence masks for encoder
    @param source_padded (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                    src_len = max source length, h = hidden size. 
    """
    source_msk = (source_padded != pad).unsqueeze(-2)
    return source_msk

def generate_tgt_masks(target_padded, pad):
    "Create a mask to hide padding and future words."
    target_msk = (target_padded != pad).unsqueeze(1)
    size = target_padded.size(1) # get seq_len for matrix

    nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    # Uncomment this if training in GPU is available
    #nopeak_mask = torch.autograd.Variable(torch.from_numpy(nopeak_mask) == 0).cuda()

    # Uncomment this if training is in CPU only
    nopeak_mask = torch.autograd.Variable(torch.from_numpy(nopeak_mask) == 0)

    target_mask = target_msk & nopeak_mask
    return target_mask