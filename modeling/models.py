import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class DNN(torch.nn.Module):
#     def __init__(self, input_size, output_size, hidden_units_size):
#         super(DNN, self).__init__()
#         if torch.cuda.device_count() > 0:
#             self.device = "cuda"
#         else:
#             self.device = "cpu"
#
#         self.input_size = input_size
#         self.output_size = output_size
#         self.hidden_units_size = [input_size] + hidden_units_size
#         self.layers = []
#         for idx, val in enumerate(hidden_units_size):
#             self.layers.append(1)
#
#         self.label = nn.Linear(hidden_units_size[-1], output_size)
#
#     def forward(self):
#         pass


class LSTMAttn(torch.nn.Module):
    def __init__(self, feature_size, hidden_size, output_size, bidirectional=True):
        super(LSTMAttn, self).__init__()

        if torch.cuda.device_count() > 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.feature_size = feature_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        if bidirectional:
            self.num_directions = 2
            self.out_hidden_size = hidden_size * 2
        else:
            self.num_directions = 1
            self.out_hidden_size = hidden_size

        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers=1, bidirectional=bidirectional, batch_first=True)
        self.label = nn.Linear(self.out_hidden_size, output_size)


    def attention_net(self, lstm_output, final_state):

        # hidden = (batch_size x out_hidden_size x 1)
        hidden = final_state.view(-1, self.out_hidden_size, 1)

        # attn_weights = (batch_size x seq_length)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)

        # soft_attn_weights = (batch_size x seq_length)
        soft_attn_weights = F.softmax(attn_weights, 1)

        # new_hidden_state = (batch_size x out_hidden_size)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state, soft_attn_weights


    def forward(self, input, input_lengths, batch_size=None):
        input.to(self.device)

        h_0 = torch.autograd.Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size, device=self.device))
        c_0 = torch.autograd.Variable(torch.zeros(self.num_directions, batch_size, self.hidden_size, device=self.device))

        input_packed = nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True)
        output_packed, (final_hidden_state, final_cell_state) = self.lstm(input_packed, (h_0, c_0))
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)

        output.to(self.device)
        attn_output, attn_weight = self.attention_net(output, final_hidden_state)
        out_label = self.label(attn_output)

        return out_label, attn_weight

