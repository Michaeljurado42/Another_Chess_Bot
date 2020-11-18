import pickle
import torch
import numpy as np
from uncertainty_rnn import BoardGuesserNetOnline, BoardGuesserNet

x = open("white_game_obs.pkl", "rb")
y = open("black_game_obs.pkl", "rb")

sense_list_1, truth_list_1 = pickle.load(x)
sense_list_2, truth_list_2 = pickle.load(y)

network = BoardGuesserNet()
network.load_state_dict(torch.load("rnn_model"))


pred_board = network(torch.Tensor([sense_list_1[0]]))[0]
pred_board2 = network(torch.Tensor([sense_list_2[0]]))[0]
x.close()
x.close()
