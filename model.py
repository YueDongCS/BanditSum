# coding:utf8

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

torch.manual_seed(233)


class SummaRuNNer(nn.Module):
    def __init__(self, config):
        super(SummaRuNNer, self).__init__()

        # Parameters
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.position_size = config.position_size
        self.position_dim = config.position_dim
        self.word_input_size = config.word_input_size
        self.sent_input_size = config.sent_input_size
        self.word_LSTM_hidden_units = config.word_GRU_hidden_units
        self.sent_LSTM_hidden_units = config.sent_GRU_hidden_units

        # Network
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        self.position_embedding = nn.Embedding(self.position_size, self.position_dim)

        self.word_LSTM = nn.LSTM(
            input_size=self.word_input_size,
            hidden_size=self.word_LSTM_hidden_units,
            batch_first=True,
            bidirectional=True)
        self.sent_LSTM = nn.LSTM(
            input_size=self.sent_input_size,
            hidden_size=self.sent_LSTM_hidden_units,
            batch_first=True,
            bidirectional=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(400, 100)
        self.fc2 = nn.Linear(400, 100)

        # Parameters of Classification Layer
        self.Wc = Parameter(torch.randn(1, 100))
        self.Ws = Parameter(torch.randn(100, 100))
        self.Wr = Parameter(torch.randn(100, 100))
        self.Wp = Parameter(torch.randn(1, 50))
        self.b = Parameter(torch.randn(1))

    def _avg_pooling(self, x, sequence_length):
        result = []
        for index, data in enumerate(x):
            avg_pooling = torch.mean(data[:sequence_length[index], :], dim=0)
            result.append(avg_pooling)
        return torch.cat(result, dim=0)

    def forward(self, x):  # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
        sequence_length = torch.sum(torch.sign(x), dim=1).data  # ex.=[3,2]-> size=2
        sequence_num = sequence_length.size()[0]  # ex. N sentes
        # print("seq_num", sequence_length)
        # word level LSTM
        word_features = self.word_embedding(x)  # Input: LongTensor (N, W), Output: (N, W, embedding_dim)
        word_outputs, _ = self.word_LSTM(word_features)  # output: word_outputs (N,W,h)

        # sentence level LSTM
        sent_features = self._avg_pooling(word_outputs, sequence_length)  # output:(N,h)
        sent_outputs, _ = self.sent_LSTM(sent_features.view(1, -1, self.sent_input_size))  # input (1,N,h)
        # document representation
        doc_features = self._avg_pooling(sent_outputs, [x.size(0)])  # output:(1,h)
        doc = self.tanh(self.fc1(doc_features))[:, None]
        # classifier layer
        outputs = []
        sent_outputs = sent_outputs.view(-1, 2 * self.sent_LSTM_hidden_units)

        s = Variable(torch.zeros(100, 1)).cuda()

        for position, sent_hidden in enumerate(sent_outputs):
            h = torch.transpose(self.tanh(self.fc2(sent_hidden.view(1, -1))), 0, 1)
            position_index = Variable(torch.LongTensor([[position]])).cuda()
            p = self.position_embedding(position_index).view(-1, 1)

            content = torch.mm(self.Wc, h)
            salience = torch.mm(torch.mm(h.view(1, -1), self.Ws), doc)
            novelty = -1 * torch.mm(torch.mm(h.view(1, -1), self.Wr), self.tanh(s))
            position = torch.mm(self.Wp, p)
            bias = self.b
            Prob = self.sigmoid(content + salience + novelty + position + bias)
            s = s + torch.mm(h, Prob)
            outputs.append(Prob)

        return torch.cat(outputs, dim=0)


class GruRuNNer(nn.Module):
    def __init__(self, config):
        super(GruRuNNer, self).__init__()

        # Parameters
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.position_size = config.position_size
        self.position_dim = config.position_dim
        self.word_input_size = config.word_input_size
        self.sent_input_size = config.sent_input_size
        self.word_GRU_hidden_units = config.word_GRU_hidden_units
        self.sent_GRU_hidden_units = config.sent_GRU_hidden_units

        # Network
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        self.position_embedding = nn.Embedding(self.position_size, self.position_dim)

        self.word_GRU = nn.GRU(
            input_size=self.word_input_size,
            hidden_size=self.word_GRU_hidden_units,
            batch_first=True,
            bidirectional=True)
        self.sent_GRU = nn.GRU(
            input_size=self.sent_input_size,
            hidden_size=self.sent_GRU_hidden_units,
            batch_first=True,
            bidirectional=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(400, 100)
        self.fc2 = nn.Linear(400, 100)

        # Parameters of Classification Layer
        self.Wc = Parameter(torch.randn(1, 100))
        self.Ws = Parameter(torch.randn(100, 100))
        self.Wr = Parameter(torch.randn(100, 100))
        self.Wp = Parameter(torch.randn(1, 50))
        self.b = Parameter(torch.randn(1))

    def _avg_pooling(self, x, sequence_length):
        result = []
        for index, data in enumerate(x):
            avg_pooling = torch.mean(data[:sequence_length[index], :], dim=0)
            result.append(avg_pooling)
        return torch.cat(result, dim=0)

    def forward(self, x):
        sequence_length = torch.sum(torch.sign(x), dim=1).data
        sequence_num = sequence_length.size()[0]

        # word level GRU
        word_features = self.word_embedding(x)
        word_outputs, _ = self.word_GRU(word_features)
        # sentence level GRU
        sent_features = self._avg_pooling(word_outputs, sequence_length)
        sent_outputs, _ = self.sent_GRU(sent_features.view(1, -1, self.sent_input_size))
        # document representation
        doc_features = self._avg_pooling(sent_outputs, [x.size(0)])
        doc = self.tanh(self.fc1(doc_features))[:, None]
        # classifier layer
        outputs = []
        sent_outputs = sent_outputs.view(-1, 2 * self.sent_GRU_hidden_units)

        s = Variable(torch.zeros(100, 1)).cuda()

        for position, sent_hidden in enumerate(sent_outputs):
            h = torch.transpose(self.tanh(self.fc2(sent_hidden.view(1, -1))), 0, 1)
            position_index = Variable(torch.LongTensor([[position]])).cuda()
            p = self.position_embedding(position_index).view(-1, 1)

            content = torch.mm(self.Wc, h)
            salience = torch.mm(torch.mm(h.view(1, -1), self.Ws), doc)
            novelty = -1 * torch.mm(torch.mm(h.view(1, -1), self.Wr), self.tanh(s))
            position = torch.mm(self.Wp, p)
            bias = self.b
            Prob = self.sigmoid(content + salience + novelty + position + bias)
            s = s + torch.mm(h, Prob)
            outputs.append(Prob)

        return torch.cat(outputs, dim=0)


class SimpleRNN(nn.Module):
    def __init__(self, config):
        super(SimpleRNN, self).__init__()

        # Parameters
        self.dropout = config.dropout
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.position_size = config.position_size
        self.position_dim = config.position_dim
        self.word_input_size = config.word_input_size
        self.sent_input_size = config.sent_input_size
        self.word_LSTM_hidden_units = config.word_GRU_hidden_units
        self.sent_LSTM_hidden_units = config.sent_GRU_hidden_units
        self.sent_rep = False

        # Network
        self.drop = nn.Dropout(self.dropout)

        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        self.position_embedding = nn.Embedding(self.position_size, self.position_dim)

        self.word_LSTM = nn.LSTM(
            input_size=self.word_input_size,
            hidden_size=self.word_LSTM_hidden_units,
            num_layers=2,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=True)
        self.sent_LSTM = nn.LSTM(
            input_size=self.sent_input_size,
            hidden_size=self.sent_LSTM_hidden_units,
            num_layers=2,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        if self.sent_rep:
            self.decoder = nn.Sequential(nn.Linear(self.sent_LSTM_hidden_units * 4, 100),
                                         nn.Tanh(),
                                         nn.Linear(100, 1),
                                         nn.Sigmoid())
        else:
            self.decoder = nn.Sequential(nn.Linear(self.sent_LSTM_hidden_units * 2, 100),
                                         nn.Tanh(),
                                         nn.Linear(100, 1),
                                         nn.Sigmoid())

    def _avg_pooling(self, x, sequence_length):
        result = []
        for index, data in enumerate(x):
            avg_pooling = torch.mean(data[:sequence_length[index], :], dim=0)
            result.append(avg_pooling)
        return torch.cat(result, dim=0)

    def forward(self, x):  # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
        sequence_length = torch.sum(torch.sign(x), dim=1).data  # ex.=[3,2]-> size=2
        sequence_num = sequence_length.size()[0]  # ex. N sentes

        # word level LSTM
        word_features = self.word_embedding(x)  # Input: LongTensor (N, W), Output: (N, W, embedding_dim)
        # word_features = self.drop(word_features)
        word_outputs, _ = self.word_LSTM(word_features)  # output: word_outputs (N,W,h)
        sent_features = self._avg_pooling(word_outputs, sequence_length).view(1, sequence_num,
                                                                              self.sent_input_size)  # output:(1,N,h)
        sent_features = self.drop(sent_features)

        # sentence level LSTM
        enc_output, _ = self.sent_LSTM(sent_features)
        enc_output = self.drop(enc_output)

        if self.sent_rep:
            doc_features = enc_output.mean(dim=1, keepdim=True).expand(enc_output.size())
            enc_output = torch.cat([enc_output ,doc_features], dim=-1)

        prob = self.decoder(enc_output)

        return prob.view(sequence_num, 1)


class SimpleRuNNer(nn.Module):
    def __init__(self, config):
        super(SimpleRuNNer, self).__init__()

        # Parameters
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.position_size = config.position_size
        self.position_dim = config.position_dim

        # Network
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        self.position_embedding = nn.Embedding(self.position_size, self.position_dim)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 100)

        # Parameters of Classification Layer
        self.Wc = Parameter(torch.randn(1, 100))
        self.Ws = Parameter(torch.randn(100, 100))
        self.Wr = Parameter(torch.randn(100, 100))
        self.Wp = Parameter(torch.randn(1, 50))
        self.b = Parameter(torch.randn(1))

    def _avg_pooling(self, x, sequence_length):
        result = []
        for index, data in enumerate(x):
            avg_pooling = torch.mean(data[:sequence_length[index], :], dim=0)
            result.append(avg_pooling)
        return torch.cat(result, dim=0)

    def forward(self, x):  # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
        sequence_length = torch.sum(torch.sign(x), dim=1).data  # ex.=[3,2]-> size=2
        sequence_num = sequence_length.size()[0]  # ex. N sentes
        # print("seq_num", sequence_length)
        # word level LSTM
        word_outputs = self.word_embedding(x)  # Input: LongTensor (N, W), Output: (N, W, embedding_dim) (49*30*100)
        sent_features = self._avg_pooling(word_outputs, sequence_length)  # output N*h
        sent_outputs = sent_features.view(1, sequence_num, -1)  # output:(N,h) (49*100)
        # sent_outputs  = sent_features.unsqueenze(1, -1)) #input (1,N,h)
        # document representation
        doc_features = self._avg_pooling(sent_outputs, [x.size(0)])  # output:(1,h)
        doc = self.tanh(self.fc1(doc_features))[:, None]
        # classifier layer
        outputs = []
        sent_outputs = sent_outputs.squeeze(0)

        s = Variable(torch.zeros(100, 1)).cuda()

        for position, sent_hidden in enumerate(sent_outputs):
            h = torch.transpose(self.tanh(self.fc2(sent_hidden.view(1, -1))), 0, 1)
            position_index = Variable(torch.LongTensor([[position]])).cuda()
            p = self.position_embedding(position_index).view(-1, 1)

            content = torch.mm(self.Wc, h)
            salience = torch.mm(torch.mm(h.view(1, -1), self.Ws), doc)
            novelty = -1 * torch.mm(torch.mm(h.view(1, -1), self.Wr), self.tanh(s))
            position = torch.mm(self.Wp, p)
            bias = self.b
            Prob = self.sigmoid(content + salience + novelty + position + bias)
            s = s + torch.mm(h, Prob)
            outputs.append(Prob)

        return torch.cat(outputs, dim=0)


class RNES(nn.Module):
    def __init__(self, config):
        super(RNES, self).__init__()

        # Parameters
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.out_channel = 50  # args.kernel_num
        self.kernel_sizes = range(0, 8)  # args.kernel_sizes[1,2,...,7]
        self.hidden_state = 400
        self.sent_input_size = 400

        # Network
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        self.conv = nn.ModuleList([nn.Conv1d(self.embedding_dim, self.out_channel,
                                             K * 2 + 1, padding=K) for K in self.kernel_sizes])

        # self.dropout = nn.Dropout(args.dropout)
        # reverse order LSTM
        self.sent_GRU = nn.GRU(
            input_size=self.sent_input_size,
            hidden_size=self.hidden_state,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.doc_encoder = nn.Sequential(nn.Linear(self.hidden_state * 2, self.hidden_state),
                                         nn.Tanh())

        self.decoder = nn.Sequential(nn.Linear(self.hidden_state * 4, 100),
                                     nn.Tanh(),
                                     nn.Linear(100, 1),
                                     nn.Sigmoid())

        self.redundancy = nn.Sequential(nn.Linear(self.hidden_state * 2, self.hidden_state),
                                        nn.Tanh())

    def _avg_pooling(self, x, sequence_length):
        result = []
        for index, data in enumerate(x):
            avg_pooling = torch.mean(data[:sequence_length[index], :], dim=0)
            result.append(avg_pooling)
        return torch.cat(result, dim=0)

    def forward(self, x):  # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
        sequence_length = torch.sum(torch.sign(x), dim=1).data  # ex.=[3,2]-> size=2
        sequence_num = sequence_length.size()[0]  # ex. N sentes

        # word level LSTM
        word_features = self.word_embedding(x)  # Input: LongTensor (N, W), Output: (N, W, embedding_dim)
        conv_input = word_features.transpose(1, 2)
        sent_features_list = []
        for i in self.kernel_sizes:
            sent_features_list.append(self.conv[i](conv_input))
        sent_features = torch.cat(sent_features_list, dim=1).transpose(1, 2)

        sent_features = self._avg_pooling(sent_features, sequence_length).view(1, sequence_num,
                                                                               self.sent_input_size)  # output:(1,N,h)

        # sentence level LSTM
        enc_output, _ = self.sent_GRU(sent_features)
        enc_output = enc_output.squeeze(0)

        doc_features = self.doc_encoder(enc_output.mean(dim=0))

        g = Variable(doc_features.data.new(self.hidden_state).zero_())

        prob_list = []
        sample_list = []
        for i in range(sequence_num):
            prob_i = self.decoder(torch.cat([enc_output[i], g, doc_features], dim=-1))
            sample_i = prob_i.bernoulli()
            prob_list.append(prob_i)
            g += self.redundancy(enc_output[i]) * sample_i
            sample_list.append(sample_i)

        prob = torch.cat(prob_list, dim=0)
        sample = torch.cat(sample_list, dim=0)

        return prob.view(sequence_num, 1), sample.view(sequence_num, 1)


class Refresh(nn.Module):
    def __init__(self, config):
        super(Refresh, self).__init__()

        # Parameters
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.out_channel = 50  # args.kernel_num
        self.kernel_sizes = range(0, 8)  # args.kernel_sizes[1,2,...,7]
        self.hidden_state = 400
        self.sent_input_size = 400

        # Network
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        self.conv = nn.ModuleList([nn.Conv1d(self.embedding_dim, self.out_channel,
                                             K * 2 + 1, padding=K) for K in self.kernel_sizes])

        # self.dropout = nn.Dropout(args.dropout)
        # reverse order LSTM
        self.sent_LSTM = nn.LSTM(
            input_size=self.sent_input_size,
            hidden_size=self.hidden_state,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.doc_encoder = nn.LSTM(
            input_size=self.sent_input_size,
            hidden_size=self.hidden_state,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.decoder = nn.Sequential(nn.Linear(self.hidden_state, 100),
                                     nn.Tanh(),
                                     nn.Linear(100, 1),
                                     nn.Sigmoid())

    def _avg_pooling(self, x, sequence_length):
        result = []
        for index, data in enumerate(x):
            avg_pooling = torch.mean(data[:sequence_length[index], :], dim=0)
            result.append(avg_pooling)
        return torch.stack(result, dim=0)

    def forward(self, x):  # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
        sequence_length = torch.sum(torch.sign(x), dim=1).data  # ex.=[3,2]-> size=2
        sequence_num = sequence_length.size()[0]  # ex. N sentes

        # word level LSTM
        word_features = self.word_embedding(x)  # Input: LongTensor (N, W), Output: (N, W, embedding_dim)
        conv_input = word_features.transpose(1, 2)
        sent_features_list = []
        for i in self.kernel_sizes:
            sent_features_list.append(self.conv[i](conv_input))
        sent_features = torch.cat(sent_features_list, dim=1).transpose(1, 2)

        sent_features = self._avg_pooling(sent_features, sequence_length)  # output:(N,h)

        # sentence level LSTM
        idx = [i for i in range(sent_features.size(0) - 1, -1, -1)]
        # idx = torch.LongTensor(idx)
        _, doc_features = self.doc_encoder(sent_features[idx].view(1, sequence_num, self.sent_input_size))

        h, _ = self.sent_LSTM(sent_features.view(1, sequence_num, self.sent_input_size), doc_features)

        prob = self.decoder(h)

        return prob.view(sequence_num, 1)


class simpleCONV(nn.Module):
    def __init__(self, config):
        super(simpleCONV, self).__init__()

        # Parameters
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.position_size = config.position_size
        self.position_dim = config.position_dim
        self.word_input_size = config.word_input_size
        self.sent_input_size = config.sent_input_size
        self.word_LSTM_hidden_units = config.word_GRU_hidden_units
        self.sent_LSTM_hidden_units = config.sent_GRU_hidden_units
        self.kernel_sizes = range(0, 8)  # args.kernel_sizes[1,2,...,7]
        self.out_channel = 50

        # Network
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))

        self.conv = nn.ModuleList([nn.Conv1d(self.embedding_dim, self.out_channel,
                                             K * 2 + 1, padding=K) for K in self.kernel_sizes])
        self.sent_LSTM = nn.LSTM(
            input_size=400,
            hidden_size=self.sent_LSTM_hidden_units,
            num_layers=2,
            batch_first=True,
            bidirectional=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.decoder = nn.Sequential(nn.Linear(self.sent_LSTM_hidden_units * 2, 100),
                                     nn.Tanh(),
                                     nn.Linear(100, 1),
                                     nn.Sigmoid())

    def _avg_pooling(self, x, sequence_length):
        result = []
        for index, data in enumerate(x):
            avg_pooling = torch.mean(data[:sequence_length[index], :], dim=0)
            result.append(avg_pooling)
        return torch.stack(result, dim=0)

    def forward(self, x):  # list of tokens ex.x=[[1,2,1],[1,1]] x = Variable(torch.from_numpy(x)).cuda()
        sequence_length = torch.sum(torch.sign(x), dim=1).data  # ex.=[3,2]-> size=2
        sequence_num = sequence_length.size()[0]  # ex. N sentes

        # word level LSTM
        word_features = self.word_embedding(x)  # Input: LongTensor (N, W), Output: (N, W, embedding_dim)
        conv_input = word_features.transpose(1, 2)
        sent_features_list = []
        for i in self.kernel_sizes:
            sent_features_list.append(self.conv[i](conv_input))
        sent_features = torch.cat(sent_features_list, dim=1).transpose(1, 2)
        sent_features = self._avg_pooling(sent_features, sequence_length).view(1, sequence_num,
                                                                               400)  # output:(1,N,h)

        # sentence level LSTM
        enc_output, _ = self.sent_LSTM(sent_features)

        prob = self.decoder(enc_output)

        return prob.view(sequence_num, 1)
