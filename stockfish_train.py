import numpy as np
import chess
from game import *
from play_game import *
from gameapi import GameAPI
from mcts import MCTS
from rbmcnet import NNetWrapper
from pickle import Pickler, Unpickler
from stockfish import Stockfish
import sys
import os
import math
import csv
import pandas as pd
from fen_string_convert import convert_fen_string

from torch.utils.data import Dataset, DataLoader



gameapi = GameAPI(chess.Board())


training_examples = []


nnet = NNetWrapper()
nnet.load_checkpoint()


loss = nnet.train("training_examples/train")

if math.isnan(loss):
   print("Is nan, not saving net")
else:
   nnet.save_checkpoint(filename="final.pth.tar")


