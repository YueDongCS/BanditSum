import pickle

from dataLoader import Vocab

print('Indexing word vectors.')

root_path = "../data/CNN_DM_pickle_data/"
glove_file = "vocab_100d.txt"
vocab_file = "vocab"

glove_vocab = Vocab()
glove_vocab.add_vocab(vocab_file=root_path + vocab_file)
glove_vocab.add_embedding(gloveFile=root_path + glove_file)
# print(glove_vocab.word_list)

pickle.dump(glove_vocab, open(root_path + 'vocab_100d.p', 'wb'))
