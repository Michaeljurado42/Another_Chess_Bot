#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:18:54 2020

@author: florianpesce
"""

import numpy as np

# import sys

position_converter = {'R': 0, 'N': 1, 'B': 2, 'Q': 3, 'K': 4, 'P': 5, 'r': 6, 'n': 7, 'b': 8, 'q': 9, 'k': 10, 'p': 11}


# one hot encoding of the fen string
def convert_fen_string(fen):
    output = np.zeros((20, 8, 8))

    split_fen = fen.split(" ")

    if (len(split_fen) != 6):
        print("Error: some fields in the FEN string are missing.")
        return

        ############################ Positions ############################
    # first dimension is for pieces. Begin with white pieces, pawn, knight, bishop, rook, queen, king. Example: blackte bishop = position 8
    # second dimension is the chess board row. So, from 1 to 8. (reverse ordering compared to fen string)
    # third dimension is the chess board column, so from a to h
    # position_converter = {'R': 0, 'N': 1, 'B': 2, 'Q': 3, 'K': 4, 'P': 5, 'r': 6, 'n': 7, 'b': 8, 'q': 9, 'k': 10, 'p': 11}

    index_row = 0
    index_column = 0
    for value in split_fen[0]:

        if (value == '/'):
            index_column = 0
            index_row += 1
            continue

        if (value.isdigit()):
            index_column += int(value)
        else:
            index_piece = position_converter[value]
            output[index_piece, index_row, index_column] = 1
            index_column += 1

    ############################ Next to play ############################
    # first index is white, second index is black

    if (split_fen[1] == 'b'):
        output[12].fill(1)

    ############################ Castling availability ############################
    # queen and then king, white and then black. 1 is for castle availability

    castling_converter = {'Q': 0, 'K': 1, 'q': 2, 'k': 3}
    if (split_fen[2] != '-'):
        for value in split_fen[2]:
            output[13 + castling_converter[value]].fill(1)

    ############################ En passant availability ############################
    # 1 if a certain square is eligible to en passant capturing
    column_converter = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}

    if (split_fen[3] != '-'):
        output[17, 7 - (int(split_fen[3][1]) - 1), column_converter[split_fen[3][0]]] = 1

    ############################ Halfmove clock ############################ 
    halfmove_clock = int(split_fen[4])
    output[18].fill(halfmove_clock)

    ############################ Fullmove number ############################ 
    fullmove_number = int(split_fen[5])
    output[19].fill(fullmove_number)

    # Dimensions: 20x8x8
    # Values: 12 for positions, 1 for next to play, 2 for white castling, 2 for black castling, 1 for en passant, 1 for halfmove_clock, 1 for fullmove_number
    return output


# np.set_printoptions(threshold= sys.maxsize)
# print(convert_fen_string("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"))


def process_sense(sense_result):
    """
    Returns the sense result as a one hot encoding in the same format as the state
    :param sense_result:
    :return:
    """
    output = np.zeros((20, 8, 8))
    for loc, piece in sense_result:
        if piece is not None:  # love how python is like english lol
            col = (loc - 1) % 8
            row = loc // 8

            piece_pos = position_converter[str(piece)]

            output[piece_pos, row, col] = 1

    return output
