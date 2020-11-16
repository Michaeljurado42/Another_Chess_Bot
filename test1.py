#%%
import numpy as np
import chess
from game import *
from play_game import *
from gameapi import GameAPI

# %%

game = Game()
# %%

board = game.truth_board
# %%
fen1 = '7k/8/8/8/8/8/8/K7 w - - 0 74'

board.set_fen(fen1)

initial_state = board._board_state

# %%

format_print_board(board)
# %%

# Print list of legal moves
# not sure what's the difference between this api and 
# board.generate_pseudo_legal_moves()
moves =list(board.generate_legal_moves())
for move in moves:
    print(move.uci(), end=' ')

len(moves)


# %%
mcts_game = GameAPI(board)

# %%

mcts_game.stringRepresentation()
# %%
valid_moves = mcts_game.getValidMoves()

#%%

np.nonzero(np.array(valid_moves))

# should be equal to [7, 14, 35, 4672]

# %%

# Test 2
fen1 = '7k/8/8/8/8/8/6K1/8 w - - 0 74'

board.set_fen(fen1)
format_print_board(board)
moves =list(board.generate_legal_moves())
for move in moves:
    print(move.uci(), end=' ')

len(moves)


mcts_game = GameAPI(board)
mcts_game.stringRepresentation()



# %%
valid_moves = mcts_game.getValidMoves()
values = np.nonzero(np.array(valid_moves))
## should get set   ([3619, 3591, 3612, 3598, 3584, 3626, 3577, 3605])
values


# %%
# Test 3
fen1 = '7k/1P6/8/8/8/8/8/8 w - - 0 74'

board.set_fen(fen1)
format_print_board(board)
moves =list(board.generate_legal_moves())
for move in moves:
    print(move.uci(), end=' ')

len(moves)


mcts_game = GameAPI(board)
mcts_game.stringRepresentation()


# %%
valid_moves = mcts_game.getValidMoves()
values = np.nonzero(np.array(valid_moves))

values
# %%

# tests for make_move
actions = [0] * 4673
actions[1029] = 1
actions = np.array(actions)

fen1 = '7k/1P6/8/8/8/8/8/8 w - - 0 74'

board.set_fen(fen1)
format_print_board(board)
moves =list(board.generate_legal_moves())
for move in moves:
    print(move.uci(), end=' ')

len(moves)

mcts_game = GameAPI(board)
mcts_game.stringRepresentation()
move = mcts_game.make_move(actions)
mcts_game.print_board()
move

actions2 = mcts_game.getValidMoves([move])

actions[4672] = 1

assert(np.sum(actions == actions2) == 4673)



#%%
np.nonzero(actions)






# %%


# %%
