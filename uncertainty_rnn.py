import torch
from torch import nn


class BoardGuesserNet(nn.Module):
    """
    This is a network for the original emission matrix 19 channels.
    For a description of the channels please see the comments under create_blank_emission_matrix in fen_string_convert.py
    """

    def __init__(self):
        super(BoardGuesserNet, self).__init__()

        """This could be a small backbone"""
        self.conv1 = torch.nn.Conv2d(18, 32, 3)
        torch.nn.init.xavier_uniform(self.conv1.weight, gain = 1)
        self.relu1 = torch.nn.LeakyReLU()
        self.pool1 = torch.nn.MaxPool2d(2)

        self.conv2 = torch.nn.Conv2d(32, 64, 2)
        torch.nn.init.xavier_uniform(self.conv2.weight, gain = 1)
        self.relu2 = torch.nn.LeakyReLU()
        self.pool2 = torch.nn.MaxPool2d(2)

        self.flatten = torch.nn.Flatten()
        # two hidden lstm states
        self.lstm = torch.nn.LSTM(256, 256, 2, batch_first=True)

        # recast board to truth
        self.dense1 = torch.nn.Linear(256, 640)
        torch.nn.init.xavier_uniform(self.dense1.weight, gain = 1)
        self.relu3 = torch.nn.LeakyReLU()

        self.dense2 = torch.nn.Linear(640, 36 * 65)
        # https://discuss.pytorch.org/t/do-i-need-to-use-softmax-before-nn-crossentropyloss/16739
#        self.softmax = torch.nn.Softmax(axis = 1)  # we use a sigmoid because it has a range of 0 to 1

    def forward(self, state: torch.Tensor):
        """

        :param state: <N, 19, 8, 8> tensor representing one game :return: <19, 36, 64> . For the encoding of the
        output please see get_truncated_truth_board in fen_string_convert.py
        :
        """
        conv_1_out = self.conv1(state)

        relu_1 = self.relu1(conv_1_out)

        conv_2_out = self.conv2(relu_1)

        relu_2 = self.relu2(conv_2_out)
        pool_2_out = self.pool2(relu_2)

        flatten_out = self.flatten(pool_2_out)
        flatten_out_batch_size_1 = torch.unsqueeze(flatten_out, 0)  # sequence length is now the number of games
        h_0, c_0 = self.lstm(
            flatten_out_batch_size_1)  # discard c_0 but we will definitely need it when we deploy the model
        remove_batch_dimension = h_0.squeeze(0)

        dense1_out = self.dense1(remove_batch_dimension)
        relu_3 = self.relu3(dense1_out)

        dense2_out = self.dense2(relu_3)

        return dense2_out.reshape((state.shape[0], 36, 65))  # apparently no softmax needed https://discuss.pytorch.org/t/do-i-need-to-use-softmax-before-nn-crossentropyloss/16739


class BoardGuesserNetOnline(nn.Module):

    def __init__(self):
        super(BoardGuesserNetOnline, self).__init__()

        """This could be a small backbone"""
        self.conv1 = torch.nn.Conv2d(18, 32, 3)
        torch.nn.init.xavier_uniform(self.conv1.weight, gain=1)
        self.relu1 = torch.nn.LeakyReLU()
        self.pool1 = torch.nn.MaxPool2d(2)

        self.conv2 = torch.nn.Conv2d(32, 64, 2)
        torch.nn.init.xavier_uniform(self.conv2.weight, gain=1)
        self.relu2 = torch.nn.LeakyReLU()
        self.pool2 = torch.nn.MaxPool2d(2)

        self.flatten = torch.nn.Flatten()
        # two hidden lstm states
        self.lstm = torch.nn.LSTM(256, 256, 2, batch_first=True)

        # recast board to truth
        self.dense1 = torch.nn.Linear(256, 640)
        torch.nn.init.xavier_uniform(self.dense1.weight, gain=1)
        self.relu3 = torch.nn.LeakyReLU()

        self.dense2 = torch.nn.Linear(640, 36 * 65)
        # https://discuss.pytorch.org/t/do-i-need-to-use-softmax-before-nn-crossentropyloss/16739

    #        self.softmax = torch.nn.Softmax(axis = 1)  # we use a sigmoid because it has a range of 0 to 1

    def forward(self, state, hidden=None):
        """

        :param state:
        :param hidden:
        :return:
        """

        """

        :param state: <N, 19, 8, 8> tensor representing one game :return: <19, 36, 64> . For the encoding of the
        output please see get_truncated_truth_board in fen_string_convert.py
        :
        """
        conv_1_out = self.conv1(state)

        relu_1 = self.relu1(conv_1_out)

        conv_2_out = self.conv2(relu_1)

        relu_2 = self.relu2(conv_2_out)
        pool_2_out = self.pool2(relu_2)

        flatten_out = self.flatten(pool_2_out)
        flatten_out_batch_size_1 = torch.unsqueeze(flatten_out, 0)  # sequence length is now the number of games
        if hidden is None:
            output, (h_0, c_0) = self.lstm(
                flatten_out_batch_size_1)  # discard c_0 but we will definitely need it when we deploy the model
        else:
            output, (h_0, c_0) = self.lstm(
                flatten_out_batch_size_1, hidden)  # discard c_0 but we will definitely need it when we deploy the model
        remove_batch_dimension = output.squeeze(0)

        dense1_out = self.dense1(remove_batch_dimension)
        relu_3 = self.relu3(dense1_out)

        dense2_out = self.dense2(relu_3)

        return dense2_out.reshape((state.shape[0], 36, 65)), (h_0, c_0)  # apparently no softmax needed https://discuss.pytorch.org/t/do-i-need-to-use-softmax-before-nn-crossentropyloss/16739





class BoardGuesserNetSlim(nn.Module):
    """
    Network that we can only use when we are predicting just the opponents pieces.
    """

    def __init__(self):
        super(BoardGuesserNet, self).__init__()

        """This could be a small backbone"""
        self.conv1 = torch.nn.Conv2d(18, 32, 3)  # use modified emission matrix instead here
        torch.nn.init.xavier_uniform(self.conv1.weight, gain = 1)
        self.relu1 = torch.nn.LeakyReLU()
        self.pool1 = torch.nn.MaxPool2d(2)

        self.conv2 = torch.nn.Conv2d(32, 64, 2)
        torch.nn.init.xavier_uniform(self.conv2.weight, gain = 1)
        self.relu2 = torch.nn.LeakyReLU()
        self.pool2 = torch.nn.MaxPool2d(2)

        self.flatten = torch.nn.Flatten()
        # two hidden lstm states
        self.lstm = torch.nn.LSTM(256, 256, 2, batch_first=True)

        # recast board to truth
        self.dense1 = torch.nn.Linear(256, 640)
        torch.nn.init.xavier_uniform(self.dense1.weight, gain = 1)
        self.relu3 = torch.nn.LeakyReLU()

        self.truth_dense = torch.nn.Linear(640, 18 * 8 * 8)

#        self.
    def forward(self, state):
        conv_1_out = self.conv1(state)

        relu_1 = self.relu1(conv_1_out)

        conv_2_out = self.conv2(relu_1)

        relu_2 = self.relu2(conv_2_out)
        pool_2_out = self.pool2(relu_2)

        flatten_out = self.flatten(pool_2_out)
        flatten_out_batch_size_1 = torch.unsqueeze(flatten_out, 0)  # sequence length is now the number of games
        h_0, c_0 = self.lstm(
            flatten_out_batch_size_1)  # discard c_0 but we will definitely need it when we deploy the model
        remove_batch_dimension = h_0.squeeze(0)

        dense1_out = self.dense1(remove_batch_dimension)
        relu_3 = self.relu3(dense1_out)

        dense2_out = self.dense2(relu_3)

        return dense2_out.reshape((state.shape[0], 18, 8, 8))  # 0 to 1



if __name__ == "__main__":

    # Test initialize RNN
    guessNet = BoardGuesserNet()
    network_input = torch.zeros((50, 19, 8, 8))
    network_guess = guessNet(network_input)

    assert network_guess.shape == (50, 18, 8, 8)
    torch.save(guessNet.state_dict(), "test_model")
    network_input_online = torch.zeros((50, 19, 8, 8))

    # test online network. With the online network we have to manually pass in hidden states
    guessNetOnline = BoardGuesserNetOnline()
    guessNetOnline.load_state_dict(torch.load("test_model"))
    truth_board, hidden_state = guessNetOnline(network_input_online)

    truth_board2, hidden_state2 = guessNetOnline(network_input_online, hidden_state)
    assert truth_board2.shape == (50, 18, 8, 8)

