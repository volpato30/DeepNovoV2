import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import deepnovo_config
from enum import Enum


activation_func = F.relu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_units = deepnovo_config.num_units


class TNet(nn.Module):
    """
    the T-net structure in the Point Net paper
    """
    def __init__(self, with_lstm=False):
        super(TNet, self).__init__()
        self.with_lstm = with_lstm
        self.conv1 = nn.Conv1d(deepnovo_config.vocab_size * deepnovo_config.num_ion + 1, num_units, 1)
        self.conv2 = nn.Conv1d(num_units, 2*num_units, 1)
        self.conv3 = nn.Conv1d(2*num_units, 4*num_units, 1)
        self.fc1 = nn.Linear(4*num_units, 2*num_units)
        self.fc2 = nn.Linear(2*num_units, num_units)
        if not with_lstm:
            self.output_layer = nn.Linear(num_units, deepnovo_config.vocab_size)
        self.relu = nn.ReLU()

        self.input_batch_norm = nn.BatchNorm1d(deepnovo_config.vocab_size * deepnovo_config.num_ion + 1)

        self.bn1 = nn.BatchNorm1d(num_units)
        self.bn2 = nn.BatchNorm1d(2*num_units)
        self.bn3 = nn.BatchNorm1d(4*num_units)
        self.bn4 = nn.BatchNorm1d(2*num_units)
        self.bn5 = nn.BatchNorm1d(num_units)

    def forward(self, x):
        """

        :param x: [batch * T, 26*8+1, N]
        :return:
            logit: [batch * T, 26]
        """
        x = self.input_batch_norm(x)
        x = activation_func(self.bn1(self.conv1(x)))
        x = activation_func(self.bn2(self.conv2(x)))
        x = activation_func(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, dim=2)  # global max pooling
        assert x.size(1) == 4*num_units

        x = activation_func(self.bn4(self.fc1(x)))
        x = activation_func(self.bn5(self.fc2(x)))
        if not self.with_lstm:
            x = self.output_layer(x)  # [batch * T, 26]
        return x


class DeepNovoPointNet(nn.Module):
    def __init__(self):
        super(DeepNovoPointNet, self).__init__()
        self.t_net = TNet(with_lstm=False)
        self.distance_scale_factor = deepnovo_config.distance_scale_factor

    def forward(self, location_index, peaks_location, peaks_intensity):
        """

        :param location_index: [batch, T, 26, 8] long
        :param peaks_location: [batch, N] N stands for MAX_NUM_PEAK, long
        :param peaks_intensity: [batch, N], float32
        :return:
            logits: [batch, T, 26]
        """

        N = peaks_location.size(1)
        assert N == peaks_intensity.size(1)
        batch_size, T, vocab_size, num_ion = location_index.size()

        peaks_location = peaks_location.view(batch_size, 1, N, 1)
        peaks_intensity = peaks_intensity.view(batch_size, 1, N, 1)
        peaks_location = peaks_location.expand(-1, T, -1, -1)  # [batch, T, N, 1]
        peaks_location_mask = (peaks_location > 1e-5).float()
        peaks_intensity = peaks_intensity.expand(-1, T, -1, -1)  # [batch, T, N, 1]

        location_index = location_index.view(batch_size, T, 1, vocab_size*num_ion)
        location_index_mask = (location_index > 1e-5).float()

        location_exp_minus_abs_diff = torch.exp(
            -torch.abs(
                (peaks_location - location_index) * self.distance_scale_factor
            )
        )
        # [batch, T, N, 26*8]

        location_exp_minus_abs_diff = location_exp_minus_abs_diff * peaks_location_mask * location_index_mask

        input_feature = torch.cat((location_exp_minus_abs_diff, peaks_intensity), dim=3)
        input_feature = input_feature.view(batch_size*T, N, vocab_size*num_ion + 1)
        input_feature = input_feature.transpose(1, 2)

        result = self.t_net(input_feature).view(batch_size, T, vocab_size)
        return result


class InitNet(nn.Module):
    def __init__(self):
        super(InitNet, self).__init__()
        self.init_state_layer = nn.Linear(deepnovo_config.embedding_size, 2 * deepnovo_config.lstm_hidden_units)

    def forward(self, spectrum_representation):
        """

        :param spectrum_representation: [N, embedding_size]
        :return:
            [num_lstm_layers, batch_size, lstm_units], [num_lstm_layers, batch_size, lstm_units],
        """
        x = torch.tanh(self.init_state_layer(spectrum_representation))
        h_0, c_0 = torch.split(x, deepnovo_config.lstm_hidden_units, dim=1)
        h_0 = torch.unsqueeze(h_0, dim=0)
        h_0 = h_0.repeat(deepnovo_config.num_lstm_layers, 1, 1)
        c_0 = torch.unsqueeze(c_0, dim=0)
        c_0 = c_0.repeat(deepnovo_config.num_lstm_layers, 1, 1)
        return h_0, c_0


class DeepNovoPointNetWithLSTM(nn.Module):
    def __init__(self):
        super(DeepNovoPointNetWithLSTM, self).__init__()
        self.t_net = TNet(with_lstm=True)
        self.embedding = nn.Embedding(num_embeddings=deepnovo_config.vocab_size,
                                      embedding_dim=deepnovo_config.embedding_size)
        self.lstm = nn.LSTM(deepnovo_config.embedding_size,
                            deepnovo_config.lstm_hidden_units,
                            num_layers=deepnovo_config.num_lstm_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(deepnovo_config.dropout_rate)
        self.output_layer = nn.Linear(deepnovo_config.num_units + deepnovo_config.lstm_hidden_units,
                                      deepnovo_config.vocab_size)

    def forward(self, location_index, peaks_location, peaks_intensity, aa_input=None, state_tuple=None):
        """

        :param location_index: [batch, T, 26, 8] long
        :param peaks_location: [batch, N] N stands for MAX_NUM_PEAK, long
        :param peaks_intensity: [batch, N], float32
        :param aa_input:[batch, T]
        :param state_tuple: (h0, c0), where each is [num_lstm_layer, batch_size, num_units] tensor
        :return:
            logits: [batch, T, 26]
        """
        assert aa_input is not None
        N = peaks_location.size(1)
        assert N == peaks_intensity.size(1)
        batch_size, T, vocab_size, num_ion = location_index.size()

        peaks_location = peaks_location.view(batch_size, 1, N, 1)
        peaks_intensity = peaks_intensity.view(batch_size, 1, N, 1)
        peaks_location = peaks_location.expand(-1, T, -1, -1)  # [batch, T, N, 1]
        peaks_location_mask = (peaks_location > 1e-5).float()
        peaks_intensity = peaks_intensity.expand(-1, T, -1, -1)  # [batch, T, N, 1]

        location_index = location_index.view(batch_size, T, 1, vocab_size * num_ion)
        location_index_mask = (location_index > 1e-5).float()

        location_exp_minus_abs_diff = torch.exp(
            -torch.abs(
                (peaks_location - location_index) * deepnovo_config.distance_scale_factor
            )
        )
        # [batch, T, N, 26*8]

        location_exp_minus_abs_diff = location_exp_minus_abs_diff * peaks_location_mask * location_index_mask

        input_feature = torch.cat((location_exp_minus_abs_diff, peaks_intensity), dim=3)
        input_feature = input_feature.view(batch_size * T, N, vocab_size * num_ion + 1)
        input_feature = input_feature.transpose(1, 2)

        ion_feature = self.t_net(input_feature).view(batch_size, T, deepnovo_config.num_units)  # attention on peaks

        # embedding
        aa_embedded = self.embedding(aa_input)
        lstm_input = aa_embedded  # [batch, T, embedding_size]
        # lstm_input = self.dropout(lstm_input)
        output_feature, new_state_tuple = self.lstm(lstm_input, state_tuple)
        output_feature = torch.cat((ion_feature, activation_func(output_feature)), dim=2)
        output_feature = self.dropout(output_feature)
        logit = self.output_layer(output_feature)
        return logit, new_state_tuple


if deepnovo_config.use_lstm:
    DeepNovoModel = DeepNovoPointNetWithLSTM
else:
    DeepNovoModel = DeepNovoPointNet


class Direction(Enum):
    forward = 1
    backward = 2


class InferenceModelWrapper(object):
    """
    a wrapper class so that the beam search part of code is the same for both with lstm and without lstm model.
    """
    def __init__(self, forward_model: DeepNovoModel, backward_model: DeepNovoModel, init_net: InitNet=None):
        self.forward_model = forward_model
        self.backward_model = backward_model
        # make sure all models are in eval mode
        self.forward_model.eval()
        self.backward_model.eval()
        if deepnovo_config.use_lstm:
            assert init_net is not None
            self.init_net = init_net
            self.init_net.eval()

    def step(self, candidate_location, peaks_location, peaks_intensity, aa_input, state_tuple, direction):
        """
        :param state_tuple: tuple of ([num_layer, batch_size, num_unit], [num_layer, batch_size, num_unit])
        :param aa_input: [batch, 1]
        :param candidate_location: [batch, 1, 26, 8]
        :param peaks_location: [batch, N]
        :param peaks_intensity: [batch, N]
        :param direction: enum class, whether forward or backward
        :return: (log_prob, new_hidden_state)
        log_prob: the pred log prob of shape [batch, 26]
        """
        if direction == Direction.forward:
            model = self.forward_model
        else:
            model = self.backward_model

        with torch.no_grad():
            if deepnovo_config.use_lstm:
                logit, new_state_tuple = model(candidate_location, peaks_location, peaks_intensity, aa_input,
                                               state_tuple)
            else:
                logit = model(candidate_location, peaks_location, peaks_intensity)
                new_state_tuple = None
            logit = torch.squeeze(logit, dim=1)
            log_prob = F.log_softmax(logit)
            #log_prob = F.logsigmoid(logit)
        return log_prob, new_state_tuple

    def initial_hidden_state(self, spectrum_representation):
        """

        :param: spectrum_representation, [batch, embedding_size]
        :return:
            [num_lstm_layers, batch_size, lstm_units], [num_lstm_layers, batch_size, lstm_units],
        """
        with torch.no_grad():
            h_0, c_0 = self.init_net(spectrum_representation)
            return h_0.to(device), c_0.to(device)


