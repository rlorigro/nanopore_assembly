from torch import nn
from torch.autograd import Variable
from torch.nn import functional as f
import torch
import numpy


COVERAGE = 50


class Encoder(nn.Module):
    def __init__(self, input_size):
        super(Encoder, self).__init__()

        self.kernel_size = (3,1)

        self.leakyrelu = nn.LeakyReLU()

        self.n_channels_0 = input_size
        self.n_channels_1 = 3
        self.n_channels_2 = 6

        self.conv2d_1 = nn.Conv2d(in_channels=self.n_channels_0, out_channels=self.n_channels_1, kernel_size=self.kernel_size, padding=(1,0))
        self.conv2d_2 = nn.Conv2d(in_channels=self.n_channels_1, out_channels=self.n_channels_2, kernel_size=self.kernel_size, padding=(1,0))

    def convolution(self, x):
        # expected convolution input shape = (batch, channel, H, W)

        # [1, 1, 50, 30]
        x = self.conv2d_1(x)
        x = self.leakyrelu(x)

        # [1, 3, 50, 30]
        x = self.conv2d_2(x)
        x = self.leakyrelu(x)

        return x

    def forward(self, x):
        # expected convolution input shape = (N, C, H, W)

        # [1, 1, 50, 30]
        x = self.convolution(x)

        # [1, 6, 50, 30]
        n, c, h, w = x.shape
        x = x.view([n,h*c,w])

        # [1, 300, 30]
        x = x.permute([0,2,1])

        return x


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout_rate):
        super(Decoder, self).__init__()

        self.rnn_input_size = input_size
        self.rnn_hidden_size = hidden_size  # aka output size
        self.rnn_n_layers = n_layers

        self.bidirectional = False
        self.n_directions = int(self.bidirectional)+1

        self.gru = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.rnn_hidden_size,
                          num_layers=self.rnn_n_layers,
                          batch_first=True,
                          dropout=dropout_rate,
                          bidirectional=self.bidirectional)

    def forward(self, x):
        # input:  (batch, seq_len, input_size)
        # hidden: (num_layers * num_directions, batch, hidden_size)

        # [1, 30, 600]
        output, h_n = self.gru(x)

        # [1, 30, 5]
        x = output.permute([0,2,1])

        # [1, 5, 30]

        return x


class EncoderDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, dropout_rate):
        super(EncoderDecoder, self).__init__()

        self.input_size = input_size
        # self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.encoder = Encoder(input_size=input_size)

        self.decoder = Decoder(input_size=300,
                               hidden_size=output_size,
                               n_layers=1,
                               dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.encoder(x)
        output = self.decoder(x)

        return output
