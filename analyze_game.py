import pickle
import torch
import numpy as np
from uncertainty_rnn import BoardGuesserNetOnline, BoardGuesserNet
from modified_play_game import play_game
from fen_string_convert import convert_truncated_to_truth, get_most_likely_truth_board

(sense_list_1, truth_list_1), (sense_list_2, truth_list_2) = play_game("random_agent_save_obs", "random_agent_save_obs")

network = BoardGuesserNetOnline()
network.load_state_dict(torch.load("black_rnn_model"))


softmax_out, hidden = network(torch.Tensor([sense_list_2[0]]), None)
print(softmax_out)
# convert it into standard truth board format
pred_board = get_most_likely_truth_board(softmax_out)
pass