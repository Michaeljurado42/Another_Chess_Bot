#%%
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

from multiprocessing import Pool
#%%

def loadTrainExamples(filename):
        examplesFile = filename + ".examples"
        if not os.path.isfile(examplesFile):
            return []
        else:
            with open(examplesFile, "rb") as f:
                training_examples = Unpickler(f).load()

        return training_examples

def save_training_examples(filename, train):
    filename = filename + ".examples"
    with open(filename, "w") as fp:
        for example in train:
            fp.write('%s,%s,%d\n' % (example[0], example[1], example[2]))


def saveTrainExamples(filename, train):
    #folder = "training_examples"

    #if not os.path.exists(folder):
    #    os.makedirs(folder)
    filename = filename + ".examples"
    with open(filename, "wb+") as f:
        Pickler(f).dump(train)
    f.closed



def play_one_game(nnet, use_stockfish):

    game = Game()
    board = game.truth_board

    move_number = 0
    game.start()
    training_examples = []

    player = random.choice([0, 1])
    use_end_game_move = True

    #print(player)
    while not game.is_over():

        move_number += 1
        
        if game.turn != player:
            #print(game.turn)
            moves = game.get_moves()
            move = random.choice(moves)
            requested_move, taken_move, captured_square, reason = game.handle_move(move)
            #format_print_board(game.truth_board)
            game.end_turn()
            continue

        current_board = game.truth_board

        gameapi = GameAPI(current_board)

        # End game move to capture opponent king if possible
        if use_end_game_move:
            move = gameapi.end_game_move(game.turn)
            if move is not None:
                pi = gameapi.getValidMoves(moves=[move])
                pi[-1] = 0
                training_examples.append([gameapi.fen, game.turn, move.uci()])
                game.handle_move(move)
                game.end_turn()
                continue

        if use_stockfish:
            # Use this to train our agent to play like stockfish
            stockfish = Stockfish("/usr/games/stockfish")
            stockfish.set_fen_position(gameapi.fen)
            move_uci = stockfish.get_best_move()
            move = chess.Move.from_uci(move_uci)
            pi = gameapi.getValidMoves(moves=[move])
            pi[-1] = 0
 
            training_examples.append([gameapi.fen, game.turn, move_uci])

        else: 
            # Use this when testing our neural network
            mcts = MCTS(gameapi, nnet, num_mcts_sims=2, cpuct=1.0)
            pi = mcts.getActionProb(temp=0)
            move = np.random.choice(len(pi), p=pi)
            
            # Collect training examples
            training_examples.append([gameapi.fen, game.turn, pi])
            move = gameapi.make_move(move, apply_move = False)

        requested_move, taken_move, captured_square, reason = game.handle_move(move)
        #format_print_board(game.truth_board)

        game.end_turn()

    #format_print_board(game.truth_board)
    win_color, win_reason = game.get_winner() 
    if player == win_color:
        result = 1
        print("You win!", move_number)
    else:
        result = -1
        print("You lost, what did you expect? ", move_number, win_reason)

    if win_color is None:
        return [(x[0], x[2], 0) for x in training_examples], 0
    else:
        return [(x[0], x[2], 1 if x[1] == win_color else -1) for x in training_examples], result


    print("Finished testing a game with", move_number, "moves")
    print("Test passed!")


def play_games_with_stockfish(filename):
    
    nnet = NNetWrapper()
    nnet.load_checkpoint()
    training_examples = []

    num_games = 4000
    for i in range(num_games):
        print("game", i, end=" ")
        examples, result = play_one_game(nnet, use_stockfish=True)
        training_examples.extend(examples)

    save_training_examples(filename, training_examples)
    return True 


def play_games_with_mcts(num_games):
    
    nnet = NNetWrapper()
    nnet.load_checkpoint(folder="models", filename="stockfish.pth.tar")

    wins = 0
    for i in range(num_games):
        print("game", i, end=" ")
        _, result = play_one_game(nnet, use_stockfish=False)
        if result == 1:
            wins += 1

    return wins



#nnet.save_checkpoint()

#training_examples = []
#for i in range(1000):
#    training_examples.extend(play_one_game(nnet))
#    saveTrainExamples(filename, training_examples)


files = ["1", "2", "3", "4", "5", "6", "7"]


# Run this with a pool of 5 agents having a chunksize of 3 until finished
agents = 7
chunksize = 1

print(sys.argv[1])

if sys.argv[1] == "train":
  with Pool(processes=agents) as pool:
   result = pool.map(play_games_with_stockfish, files, chunksize)

   training_examples = []
   #training_examples.extend(loadTrainExamples("1"))
   #training_examples.extend(loadTrainExamples("2"))
   #training_examples.extend(loadTrainExamples("3"))
   #training_examples.extend(loadTrainExamples("4"))
   #training_examples.extend(loadTrainExamples("5"))
   #training_examples.extend(loadTrainExamples("6"))
   #training_examples.extend(loadTrainExamples("7"))

   #print(len(training_examples))


   #nnet = NNetWrapper()
   #nnet.load_checkpoint()

   #loss = nnet.train(training_examples)

   #if math.isnan(loss):
   #  print("Is nan, not saving net")
   #  pass
   #else:
   #  nnet.save_checkpoint()

   #os.remove("1.examples")
   #os.remove("2.examples")
   #os.remove("3.examples")
   #os.remove("4.examples")
   #os.remove("5.examples")
   #os.remove("6.examples")
   #os.remove("7.examples")
else: 

  num_games= [10] * 7
  with Pool(processes=agents) as pool:
   result = pool.map(play_games_with_mcts, num_games, chunksize)
   print(np.sum(result)/np.sum(num_games))









