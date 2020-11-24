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


# Make move using push
# Move Rook 'R' from B8 to G8
board.push(moves[1])
format_print_board(board)

#%%

moves =list(board.generate_legal_moves())
board.push(moves[1])
format_print_board(board)

#%%




# %%
# Use pop to undo moves
board.pop()

format_print_board(board)

#%%

board.reset_board()

#%%
format_print_board(board)


# %%

board = chess.Board()
# %%
moves =list(board.generate_legal_moves())
for move in moves:
    print(move.uci(), end=' ')

# %%
board.board_fen()
# %%
board.pop()
# %%

format_print_board(board)
# %%
initial_state
# %%
moves =list(board.generate_pseudo_legal_moves())

for move in moves:
    print(move.uci(), end=' ')
print(len(moves))
# %%

format_print_board(board)
board.push(chess.Move(chess.G2, chess.F2))
format_print_board(board)
# %%

board.is_game_over()
# %%
moves =list(board.generate_pseudo_legal_moves())

for move in moves:
    print(move.uci(), end=' ')
#%%

board.push(chess.Move(chess.F5, chess.F2))
format_print_board(board)

# %%

board.is_game_over()
# %%


fen1 = '1R6/P5k1/p6p/B4r2/P1pp3P/1P1P2R1/N1P3K1/8 w - - 0 74'

board.set_fen(fen1)


# %%
format_print_board(board)
# %%
moves =list(board.generate_pseudo_legal_moves())

for move in moves:
    print(move.uci(), end=' ')

# %%

edges = set([0, 7])
for i in range(64):
    square = chess.Square(i)
    if (chess.square_rank(square) not in edges) and (chess.square_file(square) not in edges):
        print("%d %s, "%(i, chess.square_name(i)))


# %%

chess.square_file(square)
# %%

chess.square_rank(square)
# %%
