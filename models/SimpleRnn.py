from torch import nn
from torch.autograd import Variable
from torch.nn import functional as f
import torch
import numpy


COVERAGE = 50


class RnnDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size, dropout_rate, use_sigmoid=False):
        super(RnnDecoder, self).__init__()

        self.rnn_input_size = input_size
        self.rnn_hidden_size = hidden_size  # aka output size
        self.rnn_n_layers = n_layers

        self.bidirectional = True
        self.n_directions = int(self.bidirectional)+1

        self.gru = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.rnn_hidden_size,
                          num_layers=self.rnn_n_layers,
                          batch_first=True,
                          dropout=dropout_rate,
                          bidirectional=self.bidirectional)

        self.output_linear = torch.nn.Linear(hidden_size*self.n_directions, output_size)
        self.leakyrelu = torch.nn.LeakyReLU()

        self.use_sigmoid = use_sigmoid
        self.sigmoid = torch.nn.Sigmoid()

    def output_fcn(self, x):
        x = self.output_linear(x)

        if self.use_sigmoid:
            x = self.sigmoid(x)

        return x

    def output_function(self, x):
        # print("output", x.shape)
        # shape = (N, H, L) = (batch_size, hidden, length)
        # (1,16,32)

        batch_size, hidden, length = x.shape

        outputs = list()
        for l in range(length):
            x_i = x[:,:,l]
            x_i = x_i.view(batch_size, hidden, 1)
            x_i = x_i.permute([0,2,1])

            x_i = self.output_fcn(x_i)

            outputs.append(x_i)

        x = torch.cat(outputs, dim=1)

        x = x.permute([0,2,1])

        return x

    def forward(self, x):
        # input:  (batch, seq_len, input_size)
        # hidden: (num_layers * num_directions, batch, hidden_size)

        x = x.permute([0,2,1])

        output, h_n = self.gru(x)

        # [1, 30, 5]
        x = output.permute([0,2,1])

        # [1, 5, 30]
        x = self.output_function(x)

        return x


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, dropout_rate, use_sigmoid=False):
        super(Decoder, self).__init__()

        self.input_size = input_size
        # self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.decoder = RnnDecoder(input_size=input_size,
                                  hidden_size=hidden_size,
                                  n_layers=n_layers,
                                  dropout_rate=dropout_rate,
                                  output_size=output_size,
                                  use_sigmoid=use_sigmoid)

    def forward(self, x):
        # x = self.encoder(x)
        output = self.decoder(x)

        return output




