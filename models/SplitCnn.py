from torch import nn
from torch.autograd import Variable
from torch.nn import functional as f
import torch
import numpy


COVERAGE = 50


class Encoder(nn.Module):
    def __init__(self, input_size):
        super(Encoder, self).__init__()

        self.kernel_size_vertical = (3,1)
        self.kernel_size_horizontal = (1,5)

        self.leakyrelu = nn.LeakyReLU()

        self.n_channels_0_vertical = input_size
        self.n_channels_1_vertical = 16
        self.n_channels_2_vertical = 32

        self.n_channels_0_horizontal = input_size
        self.n_channels_1_horizontal = 64
        self.n_channels_2_horizontal = 128

        self.conv2d_1_vertical = nn.Conv2d(in_channels=self.n_channels_0_vertical,
                                           out_channels=self.n_channels_1_vertical,
                                           kernel_size=self.kernel_size_vertical,
                                           padding=(1,0))

        self.conv2d_2_vertical = nn.Conv2d(in_channels=self.n_channels_1_vertical,
                                           out_channels=self.n_channels_2_vertical,
                                           kernel_size=self.kernel_size_vertical,
                                           padding=(1,0))

        self.conv2d_1_horizontal = nn.Conv2d(in_channels=self.n_channels_0_horizontal,
                                             out_channels=self.n_channels_1_horizontal,
                                             kernel_size=self.kernel_size_horizontal,
                                             padding=(0,2))

        self.conv2d_2_horizontal = nn.Conv2d(in_channels=self.n_channels_1_horizontal,
                                             out_channels=self.n_channels_2_horizontal,
                                             kernel_size=self.kernel_size_horizontal,
                                             padding=(0,2))

    def vertical_convolution(self, x):
        # expected convolution input shape = (batch, channel, H, W)

        # [1, 1, 50, 30]
        x = self.conv2d_1_vertical(x)
        x = self.leakyrelu(x)

        # [1, 3, 50, 30]
        x = self.conv2d_2_vertical(x)
        x = self.leakyrelu(x)

        return x

    def horizontal_convolution(self, x):
        # expected convolution input shape = (batch, channel, H, W)

        # [1, 1, 50, 30]
        x = self.conv2d_1_horizontal(x)
        x = self.leakyrelu(x)

        # [1, 3, 50, 30]
        x = self.conv2d_2_horizontal(x)
        x = self.leakyrelu(x)

        return x

    def forward(self, x):
        # expected convolution input shape = (N, C, H, W)

        # [1, 1, 50, 30]
        x_vertical = self.vertical_convolution(x)
        x_horizontal = self.horizontal_convolution(x)

        # [1, 6, 50, 30]
        n, c, h, w = x_vertical.shape
        x_vertical = x_vertical.view([n,h*c,w])

        # [1, 6, 50, 30]
        n, c, h, w = x_horizontal.shape
        x_horizontal = x_horizontal.view([n,h*c,w])

        # print(x_vertical.shape)
        # print(x_horizontal.shape)

        x = torch.cat([x_vertical, x_horizontal], dim=1)

        # [1, 300, 30]
        # x = x.permute([0,2,1])

        return x


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size, dropout_rate):
        super(Decoder, self).__init__()

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

    def output_fcn(self, x):
        x = self.output_linear(x)

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

            x_i = self.output_linear(x_i)
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


class EncoderDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, dropout_rate):
        super(EncoderDecoder, self).__init__()

        self.encoder = Encoder(input_size=input_size)

        self.decoder = Decoder(input_size=8000,
                               output_size=output_size,
                               hidden_size=hidden_size,
                               n_layers=n_layers,
                               dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)

        return x
