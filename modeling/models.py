import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# Models for Energy Disaggregation
class LSTMAttn(torch.nn.Module):
    def __init__(self, feature_size, hidden_size, output_size, num_layers=3, bidirectional=True):
        super(LSTMAttn, self).__init__()

        if torch.cuda.device_count() > 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.feature_size = feature_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if bidirectional:
            self.num_directions = 2
            self.out_hidden_size = hidden_size * 2
        else:
            self.num_directions = 1
            self.out_hidden_size = hidden_size

        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.label = nn.Linear(self.out_hidden_size, output_size)

    def attention_net(self, lstm_output, final_state):

        # hidden = (num_layers x num_directions x batch_size x hidden_size)
        hidden = final_state.view(self.num_layers, self.num_directions, -1, self.hidden_size)

        # hidden = (num_directions x batch_size x hidden_size)
        hidden = hidden[-1].squeeze(0)

        # hidden = (batch_size x out_hidden_size x 1)
        hidden = hidden.view(-1, self.out_hidden_size, 1)

        # attn_weights = (batch_size x seq_length)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)

        # soft_attn_weights = (batch_size x seq_length)
        soft_attn_weights = F.softmax(attn_weights, 1)

        # new_hidden_state = (batch_size x out_hidden_size)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state, soft_attn_weights

    def forward(self, input, input_lengths, batch_size=None):
        input.to(self.device)

        h_0 = torch.autograd.Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=self.device))
        c_0 = torch.autograd.Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=self.device))

        input_packed = nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True)
        output_packed, (final_hidden_state, final_cell_state) = self.lstm(input_packed, (h_0, c_0))
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)

        output.to(self.device)
        attn_output, attn_weight = self.attention_net(output, final_hidden_state)
        out_label = self.label(attn_output)

        return out_label, attn_weight


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """

    #nn.init.xavier_uniform_(layer.weight)

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 3:
        (n_out, n_in, width) = layer.weight.size()
        n = n_in * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


class CNN(torch.nn.Module):
    def __init__(self, feature_size, output_size):
        super(CNN, self).__init__()

        if torch.cuda.device_count() > 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.feature_size = feature_size
        self.output_size = output_size

        self.kernel_size = (self.feature_size - 1) // 2 + 1
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(self.kernel_size, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(self.kernel_size, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.label = nn.Linear(3, self.output_size)

        self._init_layers()

    def _init_layers(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)

    def forward(self, input):
        input.to(self.device)

        input = input.view(input.shape[0], 2, input.shape[1], 1)
        input = F.relu(self.conv1(input))
        input = F.relu(self.conv2(input))
        input = F.relu(self.conv3(input))
        input = input.view(input.shape[0], input.shape[2])
        input = self.label(input)

        return input


# Models for Energy Consumption Forecasting
class LSTMForecast(torch.nn.Module):
    def __init__(self, feature_size, hidden_size, output_size, num_layers=1):
        super(LSTMForecast, self).__init__()

        if torch.cuda.device_count() > 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.feature_size = feature_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.label = nn.Linear(self.hidden_size, output_size)

    def forward(self, input, input_lengths, batch_size=None):
        input.to(self.device)

        h_0 = torch.autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device))
        c_0 = torch.autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device))

        input_packed = nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True)
        output_packed, (final_hidden_state, final_cell_state) = self.lstm(input_packed, (h_0, c_0))
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)

        output.to(self.device)
        # out_features = output[:, -1, :].squeeze()
        out_label = self.label(final_hidden_state.squeeze())

        return out_label
