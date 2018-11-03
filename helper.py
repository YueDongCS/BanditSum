# coding:utf8
import cPickle as pkl
import logging
import random
from collections import namedtuple
from copy import deepcopy

import numpy
import torch
from torch.autograd import Variable

random.seed(1234)

# os.chdir('/home/ml/ydong26/Dropbox/summarization_RL/summarization_RL/src/')
Config = namedtuple('parameters',
                    ['vocab_size', 'embedding_dim',
                     'position_size', 'position_dim',
                     'word_input_size', 'sent_input_size',
                     'word_GRU_hidden_units', 'sent_GRU_hidden_units',
                     'pretrained_embedding', 'word2id', 'id2word',
                     'dropout'])


class Document():
    def __init__(self, content, label, summary):
        self.content = content
        self.label = label
        self.summary = summary


# class Dataset():
#     def __init__(self, data_list):
#         self._data = data_list
#
#     def __len__(self):
#         return len(self._data)
#
#     def __call__(self, batch_size, shuffle=True):
#         max_len = len(self)
#         if shuffle:
#             random.shuffle(self._data)
#         batchs = [self._data[index:index + batch_size] for index in range(0, max_len, batch_size)]
#         return batchs
#
#     def __getitem__(self, index):
#         return self._data[index]


# a bunch of converter functions
def tokens_to_sentences(token_list):
    # convert a token list to sents list
    # this is a cheap fix, might need better way to do it
    if isinstance(token_list[0], list):
        sents_list = token_list
    else:
        sents_list = []
        counter = 0
        for i, token in enumerate(token_list):
            if token == '.' or token == '!' or token == '?':
                sents_list.append(token_list[counter:i + 1])  # include .!? in sents
                counter = i + 1

    sents_list = [" ".join(s) for s in sents_list]

    sents_list = [s.replace("<s>", '') for s in sents_list]
    sents_list = [s.replace("</s>", '') for s in sents_list]

    # sequence = " ".join(token_list).strip()
    # sequence = sequence.replace("\\","")
    # if "<s>" not in token_list:
    #     extra_abbreviations = ['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'i.e', 'u.s']
    #     sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #     sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)
    #     new_list = sent_tokenize(sequence)
    #     new_list = [s for s in new_list if len(s.split())>1] #all <s> and </s> is removed
    #     #print(new_list)
    # else:
    #     new_list = sequence.split("</s>")
    #     new_list = [s+"</s>" for s in new_list if len(s.split()) > 1]
    #
    #     new_list = [s.replace("<s>",'') for s in new_list]
    #     new_list = [s.replace("</s>", '') for s in new_list]
    return sents_list


def remove_control_tokens(text):
    if type(text) == str:
        text = text.replace("<s>", "")
        text = text.replace("</s>", "")
    # list of strings
    if type(text) == list:
        text = [s.replace("<s>", "") for s in text if type(s) == str]
        text = [s.replace("</s>", "") for s in text if type(s) == str]
    return text


def prepare_data(doc, word2id):
    data = deepcopy(doc.content)
    max_len = -1  # this is for padding
    for sent in data:
        words = sent.strip().split()
        max_len = max(max_len, len(words))
    sent_list = []

    for sent in data:
        words = sent.lower().strip().split()
        sent = [word2id[word] for word in words]
        if len(sent) == 0:
            continue
        sent += [0 for _ in range(max_len - len(sent))]  # this is to pad at the end of each sequence
        sent_list.append(sent)

    sent_array = numpy.array(sent_list)
    return sent_array
