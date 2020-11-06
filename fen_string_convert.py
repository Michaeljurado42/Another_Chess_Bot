#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:18:54 2020

@author: florianpesce
"""

import numpy as np

# one hot encoding of the fen string
def convert_fen_string(fen):
    
    
    split_fen = fen.split(" ")

    if (len(split_fen) != 6):
        print("Error: some fields in the FEN string are missing.")
        return 
    
    ############################ Positions ############################
    #first dimension is for pieces. Begin with white pieces, pawn, knight, bishop, rook, queen, king. Example: blackte bishop = position 8
    #second dimension is the chess board row. So, from 1 to 8. (reverse ordering compared to fen string)
    #third dimension is the chess board column, so from a to h
    positions = np.zeros((12,8,8))
    position_converter = {'R': 0, 'N': 1, 'B': 2, 'Q': 3, 'K': 4, 'P': 5, 'r': 6, 'n': 7, 'b': 8, 'q': 9, 'k': 10, 'p': 11}
    
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
            positions[index_piece, index_row, index_column] = 1
            index_column += 1
        
        
    ############################ Next to play ############################
    #first index is white, second index is black
    next_to_play = np.zeros(2)
    if (split_fen[1] == 'w'):
        next_to_play[0] = 1
    else:
        next_to_play[1] = 1
        
        
    ############################ Castling availability ############################
    # queen and then king, white and then black. 1 is for castle availability
    castling = np.zeros(4)
    
    castling_converter = {'Q': 0, 'K': 1, 'q': 2, 'k': 3}
    if (split_fen[2] != '-'):
        for value in split_fen[2]:
            castling[castling_converter[value]] = 1
            
    ############################ En passant availability ############################
    # 1 if a certain square is eligible to en passant capturing
    column_converter = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    en_passant = np.zeros((8,8))
    
    if (split_fen[3] != '-'):
        en_passant[ 7 - (int(split_fen[3][1]) - 1), column_converter[split_fen[3][0]]] = 1
        
        
    ############################ Halfmove clock ############################ 
    halfmove_clock = int(split_fen[4])
    
    
    ############################ Fullmove number ############################ 
    fullmove_number = int(split_fen[5])
    
    
    # Dimensions: 8x8x12 / 2 / 4 / 8x8 / 1 / 1
    # Values: bit / bit / bit / bit / int bounded by 100 / int
    return positions, next_to_play, castling, en_passant, halfmove_clock, fullmove_number
        

#print(convert_fen_string("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"))