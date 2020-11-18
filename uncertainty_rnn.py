import torch
from torch import nn


class BoardGuesserNet(nn.Module):

    def __init__(self):
        super(BoardGuesserNet, self).__init__()

        """This could be a small backbone"""
        self.conv1 = torch.nn.Conv2d(19, 32, 3)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2)

        self.conv2 = torch.nn.Conv2d(32, 64, 2)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2)

        self.flatten = torch.nn.Flatten()
        # two hidden lstm states
        self.lstm = torch.nn.LSTM(256, 256, 2, batch_first=True)

        # recast board to truth
        self.dense1 = torch.nn.Linear(256, 640)
        self.relu3 = torch.nn.ReLU()

        self.dense2 = torch.nn.Linear(640, 1152)
        self.sigmoid1 = torch.nn.Sigmoid()  # we use a sigmoid because it has a range of 0 to 1

    def forward(self, state):
        conv_1_out = self.conv1(state)
       # print("conv 1 out shape", conv_1_out.shape)

        relu_1 = self.relu1(conv_1_out)

        conv_2_out = self.conv2(relu_1)
      #  print("conv 2 out shape", conv_2_out.shape)

        relu_2 = self.relu2(conv_2_out)
        pool_2_out = self.pool2(relu_2)
      #  print("pool_2_out shape", pool_2_out.shape)

        flatten_out = self.flatten(pool_2_out)
        flatten_out_batch_size_1 = torch.unsqueeze(flatten_out, 0)  # sequence length is now the number of games
        h_0, c_0 = self.lstm(
            flatten_out_batch_size_1)  # discard c_0 but we will definitely need it when we deploy the model
        remove_batch_dimension = h_0.unsqueeze(0)

        dense1_out = self.dense1(flatten_out)
        relu_3 = self.relu3(dense1_out)

        dense2_out = self.dense2(relu_3)

        return self.sigmoid1(dense2_out.reshape((state.shape[0], 18, 8, 8)))  # 0 to 1


class BoardGuesserNetOnline(nn.Module):

    def __init__(self):
        super(BoardGuesserNetOnline, self).__init__()

        """This could be a small backbone"""
        self.conv1 = torch.nn.Conv2d(19, 32, 3)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2)

        self.conv2 = torch.nn.Conv2d(32, 64, 2)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2)

        self.flatten = torch.nn.Flatten()

        # two hidden lstm states
        self.lstm = torch.nn.LSTM(256, 256, 2)

        # recast board to truth
        self.dense1 = torch.nn.Linear(256, 640)
        self.relu3 = torch.nn.ReLU()

        self.dense2 = torch.nn.Linear(640, 1152)
        self.sigmoid1 = torch.nn.Sigmoid()  # we use a sigmoid because it has a range of 0 to 1

    def forward(self, state, hidden=None):
        """

        :param state:
        :param hidden:
        :return:
        """

        conv_1_out = self.conv1(state)
       # print("conv 1 out shape", conv_1_out.shape)

        relu_1 = self.relu1(conv_1_out)

        conv_2_out = self.conv2(relu_1)
       # print("conv 2 out shape", conv_2_out.shape)

        relu_2 = self.relu2(conv_2_out)
        pool_2_out = self.pool2(relu_2)
       # print("pool_2_out shape", pool_2_out.shape)

        flatten_out = self.flatten(pool_2_out)
        flatten_out_batch_size_1 = torch.unsqueeze(flatten_out, 0)  # sequence length is now the number of games

        if hidden is None:  # default hidden state is zero (is this bad)
            output, (h_0, c_0) = self.lstm(flatten_out_batch_size_1)  # discard c_0 but we will definitely
        else:
            output, (h_0, c_0) = self.lstm(flatten_out_batch_size_1, hidden)  # discard c_0 but we will definitely

        dense1_out = self.dense1(flatten_out)
        relu_3 = self.relu3(dense1_out)

        dense2_out = self.dense2(relu_3)

        return self.sigmoid1(dense2_out.reshape((state.shape[0], 18, 8, 8))), (h_0, c_0)  # 0 to 1


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

