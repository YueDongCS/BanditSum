import numpy as np
import dataLoader
import pickle as pkl
import rougefonc
from helper import tokens_to_sentences
from nltk.corpus import stopwords
###### greedily creating the label
def greedy_label_one_pair(article, summary):
    """
    :param article: a list of m sents
    :param summary: a list of n sents
    :return: a list of 0/1 labels with length m
    """
    label = np.zeros(len(article))
    x = [s.split(" ") for s in article]  # s is a string represent a sentence
    y = [s.split(" ") for s in summary]
    for i in range(len(x)):  # remove stopwords in article
        x[i] = set([w.lower() for w in x[i] if w not in stopwords.words('english')])
    for i in range(len(y)):  # remove stopwords in summary
        y[i] = set([w.lower() for w in y[i] if w not in stopwords.words('english')])

    selected_idxs = []
    for target_s in y:
        scores = [len(set(target_s).intersection(set(s))) for s in x]
        sorted_idxs = np.argsort(scores)  # index sorted in increasing order

        for i in range(1, len(sorted_idxs)):  # find none repeating best match sent
            if sorted_idxs[-i] not in selected_idxs:
                selected_idxs.append(sorted_idxs[-i])
                break

    label[selected_idxs] = 1
    # print(label)
    return label
    #
def extractive_labeling_full_dataset(dataset):
    """
    :param train_x: list of lists of sents in article
    :param train_y: list of lists of sents in gold summary
    :return: list of (dataset[i].content, dataset[i].summary, label)
    """
    data_size = len(dataset)
    print("dataset has %d data instances for greedy labelling"%data_size)
    ext_labels = []
    for i in range(data_size):
        dataset[i].content = tokens_to_sentences(dataset[i].content)
        dataset[i].summary = tokens_to_sentences(dataset[i].summary)
        label = greedy_label_one_pair(dataset[i].content, dataset[i].summary)
        ext_labels.append((dataset[i].content, dataset[i].summary, label))
        print("processed %d data out of %d" % (i, data_size))
    # store
    with open("test_ext_labels.pkl", 'wb') as f:
        pkl.dump(ext_labels, f)
    return ext_labels

def eval_greedy_labelling(dataset_with_labels):
    # evaluate
    rouge_list = []
    for data_i in dataset_with_labels:
        try:
            ref = data_i[1]
            d = zip(data_i[0], data_i[2]) #choose ext_summary
            hyp = [s[0] for s in d if s[1]==1]
            rouge_s = rougefonc.RougeTest_rouge(ref, hyp)
            rouge_list.append(rouge_s)
            # print("one example", rouge_s)
        except:
            pass
    avg_rouge = np.mean(rouge_list,axis=0)
    print("greedy_labeled_rouge", avg_rouge)
    return avg_rouge

def main():
    data_loader = dataLoader.PickleReader()
    dataset = data_loader.full_data_reader('test')
    data_with_labels = extractive_labeling_full_dataset(dataset)

    # with open("val_ext_labels.pkl", 'rb') as f:
    #     data_with_labels = pkl.load(f)
    eval_greedy_labelling(data_with_labels)


if __name__ == '__main__':
    main()