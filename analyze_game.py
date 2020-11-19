import pickle
import torch
import numpy as np
from uncertainty_rnn import BoardGuesserNetOnline, BoardGuesserNet
from modified_play_game import play_game
from fen_string_convert import convert_truncated_to_truth

(sense_list_1, truth_list_1), (sense_list_2, truth_list_2) = play_game("random_agent_save_obs", "random_agent_save_obs")

network = BoardGuesserNet()
network.load_state_dict(torch.load("rnn_model"))


softmax_out = network(torch.Tensor([sense_list_1[0]]))[0]

first_pred_label = softmax_out.detach().cpu().numpy()

# take an argmax to get the most probable board
max_pred = np.zeros(first_pred_label.shape)
max_pred[np.arange(first_pred_label.shape[0]), np.argmax(first_pred_label, axis=1)] = 1

# convert it into standard truth board format
pred_board = convert_truncated_to_truth(max_pred)
pass