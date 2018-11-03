import cPickle as pkl
import numpy

def load_dict(pre_embedding, word2id, dataset):
    count_dict = numpy.zeros(len(word2id))
    for doc in dataset:
        for sent in doc.content:
            sent = sent.strip().split()
            sent = [word2id[word] for word in sent if word in word2id]
            for w in sent:
                count_dict[w] += 1
    idx_sorted = numpy.argsort(count_dict)[::-1][:50000]

    w2id = {}
    id2word = {v: k for k, v in word2id.iteritems()}
    for i, idx in enumerate(idx_sorted):
        w2id[id2word[idx]] = i
    #this is an ugly fix for adding control tokens
    # w2id['_PAD']=0
    w2id['UNK'] = 50001
    w2id['<bos>'] = 50002
    w2id['<eos>'] = 50003
    embedding = pre_embedding[idx_sorted]
    # generating embeddings for the four control tokens
    embedding = numpy.append(embedding, embedding[1:4], axis=0)
    return embedding, w2id

if __name__ == '__main__':
    print('loading train dataset')
    train_dataset = pkl.load(open('../data/small_train.pkl'))
    # load_dict_i(train_dataset,2000)
    # train_loader = DataLoader(train_dataset)
    print("loading")
    pretrained_embedding = pkl.load(open('../data/embedding.pkl'))
    word2id = pkl.load(open('../data/word2id.pkl'))
    pretrained_embedding, word2id = load_dict(pretrained_embedding, word2id, train_dataset)
    print(len(pretrained_embedding))
