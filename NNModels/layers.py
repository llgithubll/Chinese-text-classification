from torch import nn
import torch
import torch.nn.functional as F
import copy
import numpy as np
import math
from torch.autograd import Variable


class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):
        # text = [sent len, batch size]

        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]

        embedded = embedded.permute(1, 0, 2)

        # embedded = [batch size, sent len, emb dim]

        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)

        # pooled = [batch size, embedding_dim]

        return self.fc(pooled)


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = nn.RNN(embedding_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text = [sent len, batch size]

        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]

        output, hidden = self.rnn(embedded)

        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))


import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


class TextCNN1d(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.permute(0, 2, 1)

        # embedded = [batch size, emb dim, sent len]

        conved = [F.relu(conv(embedded)) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


class BertLSTM(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [batch size, sent len]

        with torch.no_grad():
            embedded = self.bert(text)[0]

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx, batch_first=True):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=self.batch_first)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)

        # use mean
        last_tensor = torch.mean(output, dim=1)

        fc_input = self.dropout(last_tensor)
        out = self.fc(fc_input)

        return out


class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text = [batch size, sent len]

        with torch.no_grad():
            embedded = self.bert(text)[0]

        # embedded = [batch size, sent len, emb dim]

        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]

        output = self.out(hidden)

        # output = [batch size, out dim]

        return output


# class RNNModel(nn.Module):
#     """Container module with an encoder, a recurrent module, and a decoder."""
#
#     def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, attention=False,
#                  attention_width=3, cuda=False):
#         super(RNNModel, self).__init__()
#         # self.drop = nn.Dropout(dropout)
#         # self.encoder = nn.Embedding(ntoken, ninp)
#         # if rnn_type in ['LSTM', 'GRU']:
#         #     self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
#         # else:
#         #     try:
#         #         nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
#         #     except KeyError:
#         #         raise ValueError("""An invalid option for `--model` was supplied,
#         #                          options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
#         #     self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
#         if attention:
#             self.decoder = nn.Linear(nhid, ntoken)
#         else:
#             self.decoder = nn.Linear(nhid, ntoken)
#         # if tie_weights:
#         #     if nhid != ninp:
#         #         raise ValueError('When using the tied flag, nhid must be equal to emsize')
#         #     self.decoder.weight = self.encoder.weight
#
#         # self.softmax = nn.Softmax()
#         if attention:
#             self.AttentionLayer = AttentionLayer(cuda, nhid)
#         self.init_weights()
#
#         # self.rnn_type = rnn_type
#         # self.nhid = nhid
#         # self.nlayers = nlayers
#         # self.attention = attention
#         # self.attention_width = attention_width
#
#     # def init_weights(self):
#     #     initrange = 0.1
#     #     self.encoder.weight.data.uniform_(-initrange, initrange)
#     #     self.decoder.bias.data.fill_(0)
#     #     self.decoder.weight.data.uniform_(-initrange, initrange)
#     # def init_hidden(self, bsz):
#     #     weight = next(self.parameters()).data
#     #     if self.rnn_type == 'LSTM':
#     #         return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
#     #                 Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
#     #     else:
#     #         return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
#     def forward(self, input, hidden):
#         # print("input size:",input.size())
#         emb = self.drop(self.encoder(input))
#         # print("emb size:",emb.size())
#         output, hidden = self.rnn(emb, hidden)
#         # print("rnn output",output.size())
#         if self.attention:
#             output = self.AttentionLayer.forward(output, self.attention_width)
#         output = self.drop(output)
#         decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
#         return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden


class AttentionModel(torch.nn.Module):
    def __init__(self, args):
        super(AttentionModel, self).__init__()
        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : num classes
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_dim : Embeddding dimension of GloVe word embeddings
        --------
        """
        self.batch_size = args.batch_size
        self.output_size = args.output_size
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.embed_dim = args.embed_dim
        self.bidirectional = args.bidirectional
        self.dropout = args.dropout
        self.use_cuda = args.cuda
        self.sequence_length = args.sequence_length
        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=const.PAD)
        self.lookup_table.weight.data.uniform_(-1., 1.)
        self.layer_size = args.layer_size
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_size, num_layers=self.layer_size,
                            dropout=self.dropout, bidirectional=self.bidirectional)
        if self.bidirectional:
            self.num_direction = 2
        else:
            self.num_direction = 1

        self.attention_size = args.attention_size
        if self.use_cuda:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.num_direction, self.attention_size).cuda())
            self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
        else:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.num_direction, self.attention_size))
            self.u_omega = Variable(torch.zeros(self.attention_size))

        self.label = nn.Linear(self.hidden_size * self.num_direction, self.output_size)

    def attention_net(self, lstm_output):
        # lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        """
        print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)
        print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)
        print(attn_tanh.size())  (squence_length * batch_size, attention_size)
        print(attn_hidden_layer.size())  (squence_length * batch_size, 1)
        print(exps.size())  (batch_size, squence_length)
        print(alphas.size()) (batch_size, squence_length)
        print(alphas_reshape.size()) = (batch_size, squence_length, 1)
        print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)
        print(attn_output.size()) = (batch_size, hidden_size*layer_size)
        """

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.num_direction])
        # M = tanh(H)
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # alpha = softmax(omega.T*M)
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, lstm_output.size()[0]])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas, [-1, lstm_output.size()[0], 1])
        state = lstm_output.permute(1, 0, 2)
        # r = H*alpha.T
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def forward(self, input_sentences, batch_size=None):
        input = self.lookup_table(input_sentences)
        input = input.permute(1, 0, 2)

        if self.use_cuda:
            hidden_state = Variable(
                torch.zeros(self.layer_size * self.num_direction, len(input_sentences), self.hidden_size).cuda())
            cell_state = Variable(
                torch.zeros(self.layer_size * self.num_direction, len(input_sentences), self.hidden_size).cuda())
        else:
            hidden_state = Variable(
                torch.zeros(self.layer_size * self.num_direction, len(input_sentences), self.hidden_size))
            cell_state = Variable(
                torch.zeros(self.layer_size * self.num_direction, len(input_sentences), self.hidden_size))

        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        attn_output = self.attention_net(lstm_output)
        logits = self.label(attn_output)
        return logits


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx, batch_first=False,use_cuda=True,attention_size=85):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.attention_size = attention_size

        if self.bidirectional:
            self.num_direction = 2
        else:
            self.num_direction = 1

        if use_cuda:
            self.w_omega = Variable(torch.zeros(self.hidden_dim * self.num_direction, self.attention_size).cuda())
            self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
        else:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.num_direction, self.attention_size))
            self.u_omega = Variable(torch.zeros(self.attention_size))

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def attention_net(self, lstm_output):
        # lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        """
        print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)
        print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)
        print(attn_tanh.size())  (squence_length * batch_size, attention_size)
        print(attn_hidden_layer.size())  (squence_length * batch_size, 1)
        print(exps.size())  (batch_size, squence_length)
        print(alphas.size()) (batch_size, squence_length)
        print(alphas_reshape.size()) = (batch_size, squence_length, 1)
        print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)
        print(attn_output.size()) = (batch_size, hidden_size*layer_size)
        """

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_dim * self.num_direction])
        # M = tanh(H)
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # alpha = softmax(omega.T*M)
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, lstm_output.size()[0]])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas, [-1, lstm_output.size()[0], 1])
        state = lstm_output.permute(1, 0, 2)
        # r = H*alpha.T
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=self.batch_first)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        attention_output = self.attention_net(output)
        # use mean
        # last_tensor = torch.mean(attention_output, dim=1)

        fc_input = self.dropout(attention_output)
        out = self.fc(fc_input)

        return out
