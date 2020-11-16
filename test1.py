#%%
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
np.nonzero(np.array(valid_moves))

# %%
