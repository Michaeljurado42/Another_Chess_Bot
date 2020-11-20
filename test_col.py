#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:15:32 2020

@author: florianpesce
"""
import numpy as np
import chess
from fen_string_convert import process_sense, convert_fen_string, get_row_col_from_num, create_blank_emission_matrix, get_truncated_board, start_bookkeeping, find_piece_type

fen_string = "rnbqkbnr/pppppppp/8/8/8/2P5/PP1PPPPP/RNBQKBNR w KQkq - 0 1"
board = chess.Board(fen_string)
print(board)
emission_matrix = create_blank_emission_matrix(True)
bookkeeping = start_bookkeeping(True)
emission_matrix[:12] = np.copy(bookkeeping)
piece_type = 5
from_row = 1
from_col = 2
to_row = 2
to_col = 2
bookkeeping[piece_type, from_row, from_col] = 0
bookkeeping[piece_type, to_row, to_col] = 1
emission_matrix[piece_type, from_row, from_col] = 0
emission_matrix[piece_type, to_row, to_col] = 1
print("Bookkeeping")
print(bookkeeping)
print("Emission matrix")
print(emission_matrix[:12])

print("/////////////////// TEST /////////////////////")

move = chess.Move(13, 29)
from_row, from_col = get_row_col_from_num(move.from_square)
to_row, to_col = get_row_col_from_num(move.to_square)
piece_type = find_piece_type(bookkeeping,from_row,from_col)
bookkeeping[piece_type, from_row, from_col] = 0
bookkeeping[piece_type, to_row, to_col] = 1
emission_matrix[piece_type, from_row, from_col] = 0
emission_matrix[piece_type, to_row, to_col] = 1
emission_matrix[13 - 1, from_row, from_col] = 0
emission_matrix[13 - 1, to_row, to_col] = 1

board.push(move)

print(board)
print("Bookkeeping")
print(bookkeeping)
print("Emission matrix")
print(emission_matrix[:12])