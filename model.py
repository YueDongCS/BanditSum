# coding:utf8

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

torch.manual_seed(233)


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
            enc_output = torch.cat([enc_output, doc_features], dim=-1)

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
