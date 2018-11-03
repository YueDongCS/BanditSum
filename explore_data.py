import argparse
import cPickle as pkl
import logging
import random
import sys

import torch

from helper import DataLoader


def argsParser():
    torch.manual_seed(233)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')

    parser = argparse.ArgumentParser()

    parser.add_argument('--emb_file', type=str, default='../data/embedding.pkl')
    parser.add_argument('--train_file', type=str, default='../data/train.pkl')
    parser.add_argument('--validation_file', type=str, default='../data/validation.pkl')
    parser.add_argument('--test_file', type=str, default='../data/test.pkl')
    parser.add_argument('--model_file', type=str, default='../model/summary.model')
    parser.add_argument('--epochs_ml', type=int, default=5)
    parser.add_argument('--epochs_rl', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--rl_baseline', type=bool, default=False)

    args = parser.parse_args()
    return args


class DataExplore():
    def __init__(self):
        self.args = argsParser()
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    def load_train(self):
        logging.info('loadding train dataset')
        self.train_dataset = pkl.load(open(self.args.train_file))

    def load_val(self):
        logging.info('loadding validation dataset')
        self.validation_dataset = pkl.load(open(self.args.validation_file))

    def load_test(self):
        logging.info('loadding test dataset')
        self.test_dataset = pkl.load(open(self.args.test_file))

    def print_example(self, dataset):
        index = random.randrange(len(dataset))
        print(dataset[index].content)
        print(dataset[index].label)
        print(dataset[index].summary)

    def print_dataset_stats(self, dataset):
        data_loader = DataLoader(dataset, shuffle=True)
        num_of_docs = len(dataset)

        total_length = 0
        total_summary = 0
        total_reference = 0
        min_length = 1000
        max_length = 0

        for docs in data_loader:
            doc = docs[0]
            total_length += len(doc.label)
            total_summary += sum(doc.label)
            total_reference += len(doc.summary)

            if len(doc.label) < min_length:
                min_length = len(doc.label)
            if len(doc.label) > max_length:
                max_length = len(doc.label)

        print 'total docs in:' + str(num_of_docs)
        print 'avg_num_of_sentences:    ' + str(total_length * 1.0 / num_of_docs)
        print 'avg_num_of_sentences_in_summary:   ' + str(total_summary * 1.0 / num_of_docs)
        print 'avg_num_of_sentences_in_reference: ' + str(total_reference * 1.0 / num_of_docs)
        print 'min_num_of_sentences_in_data is:' + str(min_length)
        print 'max_num_of_sentences_in_data is:' + str(max_length)
        sys.stdout.flush()


def main():
    data_explore = DataExplore()

    data_explore.load_train()
    data_explore.print_dataset_stats(data_explore.train_dataset)

    data_explore.load_test()
    data_explore.print_dataset_stats(data_explore.test_dataset)

    data_explore.load_val()
    data_explore.print_dataset_stats(data_explore.validation_dataset)


if __name__ == '__main__':
    main()
