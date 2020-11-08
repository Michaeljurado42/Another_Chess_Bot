#!/usr/bin/env python

#%%
import chess
from game import *
from play_game import *
# %%

game = Game()
# %%

board = game.truth_board
# %%
fen1 = '1R6/6k1/p6p/B4r2/P1pp3P/1P1P2R1/N1P3K1/8 w - - 0 74'

board.set_fen(fen1)


# %%

format_print_board(board)
# %%

# Print list of legal moves
# not sure what's the difference between this api and 
# board.generate_pseudo_legal_moves()
moves =list(board.generate_legal_moves())
for move in moves:
    print(move.uci(), end=' ')
# %%


# Make move using push
# Move Rook 'R' from B8 to G8
board.push(moves[1])
format_print_board(board)




# %%
# Use pop to undo moves
board.pop()

format_print_board(board)
# %%
