import hashlib
import os
import pickle, tarfile
import random

import numpy as np

PAD_ID = 0
UNK_ID = 1
wordembed_size = 200

human_eval_set = [
    '00180b7ce54794a52766d795506a94071f7c055b',
    '00a51d5454f2ef7dbf4c53471223a27fb9c20681',
    '1a9fb7c99082836c6a41091f38c1914c51c34e4f',
    '2dfa0295b48249b24c373af7319a1b3ec027a549',
    '3074bd292f4c218ac90b24e703944365bf1088a0',
    '4523ba72ce198cb004dcca42c5c5af092e4fffcc',
    '4b5becaee812ea2300dabf1bb3b11bab7263c8eb',
    '5776732bfe072fcac0a9cbe14162992255d0ad26',
    '5c09b29c6b6b147188a03c9d41cdae712898034c',
    '6528057a6759349f1fb146da9e553d7d38625f21',
    '78303d514399582305d21c8c92b0e57f7d254949',
    '79c8c2925651b57c8c802dc96a1f87877b1c765d',
    '89d1c32caf60a8b9b73f0ef6a3c34033fada9c1d',
    '8ba20bec4358b39f84a8a07264f71566bb3c5e8e',
    '8c2071e749ae4dbbeb5cffe4c87abbd075fd98fd',
    '8cd4ce0d79ba06ed59d743d70ccf8bab9308cdd6',
    'd1fa0db909ce45fe1ee32d6cbb546e9d784bcf74',
    'dae2675302d92bdf0bbd6d35c3e473389f8bb5a1',
    'dca50abe4ea90250a2b709816cde88c974a9e3fd',
    'f9ff3271266864347d4c612ea485d3dd8fb63543',
]


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  # acceptable ways to end a sentence


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    # print line[-1]
    return line + " ."


class Document:
    def __init__(self, content, summary):
        self.content = content
        self.summary = summary


class Dataset:
    def __init__(self, data_list):
        self._data = data_list

    def __len__(self):
        return len(self._data)

    def __call__(self, batch_size, shuffle=True):
        max_len = len(self)
        if shuffle:
            random.shuffle(self._data)
        batchs = [self._data[index:index + batch_size] for index in range(0, max_len, batch_size)]
        return batchs

    def __getitem__(self, index):
        return self._data[index]


class Vocab:
    def __init__(self):
        self.word_list = ['<pad>', '<unk>', '<s>', '<\s>']
        self.w2i = {}
        self.i2w = {}
        self.count = 0
        self.embedding = None

    def __getitem__(self, key):
        if self.w2i.has_key(key):
            return self.w2i[key]
        else:
            return self.w2i['<unk>']

    def add_vocab(self, vocab_file="../data/finished_files/vocab"):
        with open(vocab_file, "rb") as f:
            for line in f:
                self.word_list.append(line.split()[0])  # only want the word, not the count
        print("read %d words from vocab file" % len(self.word_list))

        for w in self.word_list:
            self.w2i[w] = self.count
            self.i2w[self.count] = w
            self.count += 1

    def add_embedding(self, gloveFile="../data/finished_files/glove.6B/glove.6B.100d.txt", embed_size=100):
        print("Loading Glove embeddings")
        with open(gloveFile, 'r') as f:
            model = {}
            w_set = set(self.word_list)
            embedding_matrix = np.zeros(shape=(len(self.word_list), embed_size))

            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                if word in w_set:  # only extract embeddings in the word_list
                    embedding = np.array([float(val) for val in splitLine[1:]])
                    model[word] = embedding
                    embedding_matrix[self.w2i[word]] = embedding
                    if len(model) % 1000 == 0:
                        print("processed %d data" % len(model))
        self.embedding = embedding_matrix
        print("%d words out of %d has embeddings in the glove file" % (len(model), len(self.word_list)))


class BatchDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=True):
        assert isinstance(dataset, Dataset)
        assert len(dataset) >= batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        return iter(self.dataset(self.batch_size, self.shuffle))


class PickleReader:
    """
    this class intends to read pickle files converted by RawReader
    """

    def __init__(self, pickle_data_dir="../data/CNN_DM_pickle_data/"):
        """
        :param pickle_data_dir: the base_dir where the pickle data are stored in
        this dir should contain train.p, val.p, test.p, and vocab.p
        this dir should also contain the chunked_data folder
        """
        self.base_dir = pickle_data_dir

    def data_reader(self, dataset_path):
        """
        :param dataset_path: path for data.p
        :return: data: Dataset objects (contain Document objects with doc.content and doc.summary)
        """
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)
        return data

    def full_data_reader(self, dataset_type="train"):
        """
        this method read the full dataset
        :param dataset_type: "train", "val", or "test"
        :return: data: Dataset objects (contain Document objects with doc.content and doc.summary)
        """
        return self.data_reader(self.base_dir + dataset_type + ".p")

    def chunked_data_reader(self, dataset_type="train", data_quota=-1):
        """
        this method reads the chunked data in the chunked_data folder
        :return: a iterator of chunks of datasets
        """
        data_counter = 0
        # chunked_dir = self.base_dir + "chunked/"
        chunked_dir = os.path.join(self.base_dir, 'chunked')
        os_list = os.listdir(chunked_dir)
        if data_quota == -1:  # none-quota randomize data
            random.seed()
            random.shuffle(os_list)

        for filename in os_list:
            if filename.startswith(dataset_type):
                # print("filename:", filename)
                chunk_data = self.data_reader(os.path.join(chunked_dir, filename))
                if data_quota != -1:  # cut off applied
                    quota_left = data_quota - data_counter
                    # print("quota_left", quota_left)
                    if quota_left <= 0:  # no more quota
                        break
                    elif quota_left > 0 and quota_left < len(chunk_data):  # return partial data
                        yield Dataset(chunk_data[:quota_left])
                        break
                    else:
                        data_counter += len(chunk_data)
                        yield chunk_data
                else:
                    yield chunk_data
            else:
                continue

    def refresh_test_reader(self, eval_path):
        tar_gold = tarfile.open(os.path.join(eval_path,
                                             'Refresh-NAACL18-baseline-gold-data.tar.gz'),
                                mode='r:gz')
        gold_dict = {}
        for member in tar_gold.getmembers():
            f = tar_gold.extractfile(member)
            if f and f.name.find('gold-cnn-dailymail-test-orgcase') >= 0:
                lines = f.read()
                lines = lines.lower().strip().split('\n')
                lines = [fix_missing_period(line) for line in lines]
                # Make article into a single string
                gold = ' '.join(lines)

                # Make abstract into a signle string, putting <s> and </s> tags around the sentences
                gold = ' '.join(["%s %s %s" % ('<s>', sent, '</s>') for sent in lines])

                _, name = os.path.split(f.name)
                name = name.split('.')[0]
                gold_dict[name] = gold.split(' ')

        tar_news = tarfile.open(os.path.join(eval_path,
                                             'Refresh-NAACL18-CNN-DM-Filtered-TokenizedSegmented.tar.gz'),
                                mode='r:gz')
        news_dict = {}
        for member in tar_news.getmembers():
            f = tar_news.extractfile(member)
            if f and f.name.find('test') >= 0:
                lines = f.read()
                lines = lines.lower().strip().split('\n')
                lines = [fix_missing_period(line) for line in lines]
                # Make article into a single string
                news = ' '.join(lines)

                _, name = os.path.split(f.name)
                name = name.split('.')[0]
                news_dict[name] = news.split(' ')

        assert set(news_dict.keys()).issuperset(set(gold_dict.keys()))

        testset = []
        for k in gold_dict.keys():
            # for k in human_eval_set:
            testset.append(Document(news_dict[k], gold_dict[k]))

        return [Dataset(testset)]


def main():
    def get_art_abs(story_file):
        lines = read_text_file(story_file)

        # Lowercase everything
        lines = [line.lower() for line in lines]

        # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
        lines = [fix_missing_period(line) for line in lines]

        # Separate out article and abstract sentences
        article_lines = []
        highlights = []
        next_is_highlight = False
        for idx, line in enumerate(lines):
            if line == "":
                continue  # empty line
            elif line.startswith("@highlight"):
                next_is_highlight = True
            elif next_is_highlight:
                highlights.append(line)
            else:
                article_lines.append(line)

        # Make article into a single string
        article = ' '.join(article_lines)

        # Make abstract into a signle string, putting <s> and </s> tags around the sentences
        abstract = ' '.join(["%s %s %s" % ('<s>', sent, '</s>') for sent in highlights])

        return article.split(' '), abstract.split(' ')

    def write_to_pickle(url_file, out_file, chunk_size=1000):
        url_list = read_text_file(url_file)
        url_hashes = get_url_hashes(url_list)
        url = zip(url_list, url_hashes)
        story_fnames = ["/home/hmwv1114/workdisk/workspace/cnn_dm_stories/cnn_stories_tokenized/" + s + ".story"
                        if u.find(
            'cnn.com') >= 0 else "/home/hmwv1114/workdisk/workspace/cnn_dm_stories/dm_stories_tokenized/" + s + ".story"
                        for u, s in url]

        new_lines = []
        for i, filename in enumerate(story_fnames):
            if i % chunk_size == 0 and i > 0:
                pickle.dump(Dataset(new_lines), open(out_file % (i / chunk_size), "wb"))
                new_lines = []

            try:
                art, abs = get_art_abs(filename)
            except:
                print(filename)
                continue
            new_lines.append(Document(art, abs))

        if new_lines != []:
            pickle.dump(Dataset(new_lines), open(out_file % (i / chunk_size + 1), "wb"))

    train_urls = "../data/url_lists/all_train.txt"
    val_urls = "../data/url_lists/all_val.txt"
    test_urls = "../data/url_lists/all_test.txt"

    write_to_pickle(test_urls, "../data/CNN_DM_pickle_data/chunked/test_%03d.bin.p", chunk_size=100000000)
    write_to_pickle(val_urls, "../data/CNN_DM_pickle_data/chunked/val_%03d.bin.p", chunk_size=100000000)
    write_to_pickle(train_urls, "../data/CNN_DM_pickle_data/chunked/train_%03d.bin.p")


if __name__ == "__main__":
    # duc_reader = DucReader()
    # duc_reader.load_articles()
    main()
