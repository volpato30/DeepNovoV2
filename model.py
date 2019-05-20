import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import deepnovo_config
from enum import Enum


activation_func = F.relu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SpectrumCNN(nn.Module):
    def __init__(self, dropout_p=0.25):
        super(SpectrumCNN, self).__init__()
        kernel_size = deepnovo_config.SPECTRUM_RESOLUTION
        reduced_size = int(deepnovo_config.MZ_SIZE / kernel_size)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=kernel_size, stride=kernel_size)
        self.conv_1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=2)  # SAME padding
        self.conv_2 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5, stride=1, padding=2)  # SAME padding
        self.max_pool_2 = nn.MaxPool1d(kernel_size=6, stride=4, padding=1)  # SAME padding
        self.dropout = nn.Dropout(dropout_p)
        in_feature_size = reduced_size
        self.linear = nn.Linear(in_feature_size, deepnovo_config.num_units)
        self.output = nn.Linear(deepnovo_config.num_units, 2 * deepnovo_config.num_units)

    def forward(self, spectrum_holder):
        """
        encode the spectrum to vectors, which are used as the initial state of
        :param spectrum_holder:
        :return: (h0, c0), where each is a [batch, num_lstm_layer, num_units] tensor
        """
        batch_size, mz_size = spectrum_holder.size()
        spectrum_holder = spectrum_holder.view(batch_size, 1, mz_size)
        net = self.max_pool_1(spectrum_holder)
        net = activation_func(self.conv_1(net))
        net = activation_func(self.conv_2(net))
        net = self.max_pool_2(net)
        net = self.dropout(net)
        net = net.view(batch_size, -1)
        net = activation_func(self.linear(net))
        net = self.dropout(net)
        net = activation_func(self.output(net))
        net = net.unsqueeze(0)  # [1, batch_size, 2*num_units]
        net = net.expand(deepnovo_config.num_lstm_layers, -1, -1)
        h0, c0 = torch.split(net, deepnovo_config.num_units, dim=2)
        return h0.contiguous(), c0.contiguous()


class IonCNN(nn.Module):
    def __init__(self, dropout_p=0.25):
        super(IonCNN, self).__init__()
        self.conv_1 = nn.Conv2d(deepnovo_config.vocab_size, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_2 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.dropout = nn.Dropout(dropout_p)
        in_feature_size = int(deepnovo_config.num_ion * deepnovo_config.WINDOW_SIZE // 2 * 64)
        self.linear_1 = nn.Linear(in_feature_size, deepnovo_config.num_units)

    def forward(self, candidate_intensity):
        """

        :param candidate_intensity: [batch_size, T, 26, 8, 10]
        :return:
        ion cnn feature of size [batch_size, T, num_units]
        """
        batch_size, T, vocab_szie, num_ion, window_size = candidate_intensity.size()
        candidate_intensity = candidate_intensity.view(batch_size*T, vocab_szie, num_ion, window_size)
        net = activation_func(self.conv_1(candidate_intensity))
        net = activation_func(self.conv_2(net))
        net = activation_func(self.conv_3(net))
        net = self.maxpool(net)  # [batch_size*T, 64, 8, 5]
        net = self.dropout(net)
        net = net.view(batch_size*T, -1)
        net = activation_func(self.linear_1(net))
        net = net.view(batch_size, T, deepnovo_config.num_units)
        return net


class DeepNovoLSTM(nn.Module):
    def __init__(self, vocab_size, num_ion, num_units, dropout_rate=0.25):
        super(DeepNovoLSTM, self).__init__()
        self.num_units = num_units
        self.num_ion = num_ion
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=deepnovo_config.vocab_size,
                                      embedding_dim=deepnovo_config.num_units)
        self.lstm = nn.LSTM(deepnovo_config.embedding_size, deepnovo_config.num_units,
                            num_layers=deepnovo_config.num_lstm_layers,
                            batch_first=True)
        self.output_layer = nn.Linear(2*deepnovo_config.num_units, deepnovo_config.vocab_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.ion_cnn = IonCNN()

    def forward(self, candidate_intensity, aa_input, state_tuple):
        """

        :param candidate_intensity: [batch_size, T, 26, 8, 10]
        :param aa_input: [batch_size, T]
        :param state_tuple: (h0, c0), where each is [num_lstm_layer, batch_size, num_units] tensor
        :return: a tuple
        logit: [batch, T, 26]
        """
        ion_cnn_feature = self.ion_cnn(candidate_intensity)  # [batch, T, num_units]
        aa_embedded = self.embedding(aa_input)
        lstm_feature, new_state_tuple = self.lstm(aa_embedded, state_tuple)  # [batch, T, num_units], (h_t, c_t)
        concat_feature = torch.cat((ion_cnn_feature, lstm_feature), dim=2)
        logit = self.output_layer(concat_feature)
        return logit, new_state_tuple


class DeepNovoWithoutLSTM(nn.Module):
    def __init__(self, vocab_size, num_ion, num_units, dropout_rate=0.25):
        super(DeepNovoWithoutLSTM, self).__init__()
        self.num_units = num_units
        self.num_ion = num_ion
        self.vocab_size = vocab_size
        self.output_layer = nn.Linear(deepnovo_config.num_units, deepnovo_config.vocab_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.ion_cnn = IonCNN()

    def forward(self, candidate_intensity, holder_1=None, holder_2=None):
        """
        has the same signature with corresponding method in WithLSTM model
        :param holder_2: just a placeholder
        :param holder_1: just a placeholder
        :param candidate_intensity: [batch_size, T, 26, 8, 10]
        :param aa_input: [batch_size, T]
        :param state_tuple: (h0, c0), where each is [num_lstm_layer, batch_size, num_units] tensor
        :return: a tuple
        logit: [batch, T, 26]
        """
        ion_cnn_feature = self.ion_cnn(candidate_intensity)  # [batch, T, num_units]
        logit = self.output_layer(ion_cnn_feature)
        return logit, None


# defalt use lstm
if deepnovo_config.use_lstm:
    DeepNovoModel = DeepNovoLSTM
else:
    DeepNovoModel = DeepNovoWithoutLSTM

class Direction(Enum):
    forward = 1
    backward = 2


class InferenceModelWrapper(object):
    """
    a wrapper class so that the beam search part of code is the same for both with lstm and without lstm model.
    """
    def __init__(self, forward_model: DeepNovoModel, backward_model: DeepNovoModel, spectrum_cnn=None):
        if spectrum_cnn is None:
            assert deepnovo_config.use_lstm == False
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.spectrum_cnn = spectrum_cnn
        # make sure all models are in eval mode
        self.forward_model.eval()
        self.backward_model.eval()
        if deepnovo_config.use_lstm:
            self.spectrum_cnn.eval()
        else:
            assert spectrum_cnn is None

    def initial_hidden_state(self, spectrum_holder_list: list):
        """
        get initial hidden state
        :param spectrum_holder: list of np.ndarray
        :return: (h0, c0), each is [num_layer, batch, num_units] tensor
        """
        if not deepnovo_config.use_lstm:
            return None
        else:
            temp = np.array(spectrum_holder_list)
            with torch.no_grad():
                spectrum_holder = torch.from_numpy(temp).to(device)
                return self.spectrum_cnn(spectrum_holder)

    def step(self, candidate_intensity, aa_input, prev_hidden_state_tuple, direction):
        """
        :param candidate_intensity: [batch, 1, 26, 8, 10]
        :param aa_input: [batch, 1]
        :param prev_hidden_state_tuple: (h, c), each is [batch, 1, num_units]
        :param direction: enum class, whether forward or backward
        :return: (log_prob, new_hidden_state)
        log_prob: the pred log prob of shape [batch, 26]
        new_hidden_state: new hidden state for next step
        """
        if direction == Direction.forward:
            model = self.forward_model
        else:
            model = self.backward_model
        if deepnovo_config.use_lstm:
            assert candidate_intensity.size(1) == aa_input.size(1) == 1
        with torch.no_grad():
            logit, new_hidden_state = model(candidate_intensity, aa_input, prev_hidden_state_tuple)
            logit = torch.squeeze(logit, dim=1)
            log_prob = F.log_softmax(logit)
        return log_prob, new_hidden_state


