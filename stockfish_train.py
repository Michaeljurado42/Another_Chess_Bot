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



gameapi = GameAPI(chess.Board())

def convert_move(move_uci):
    move = chess.Move.from_uci(move_uci)
    pi = gameapi.getValidMoves(moves=[move])
    pi[-1] = 0
    return pi


training_examples = []


nnet = NNetWrapper()
nnet.load_checkpoint()


with open('training_examples/train') as csvfile:
    data = list(csv.reader(csvfile))

print(len(data))

examples = [(x[0], convert_move(x[1]), int(x[2])) for x in data]

loss = nnet.train(examples)

if math.isnan(loss):
   print("Is nan, not saving net")
else:
   nnet.save_checkpoint(filename="final.pth.tar")


