#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:18:54 2020

@author: florianpesce
"""

import numpy as np

# import sys
import torch
import chess

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

# truncated one hot encoding of the fen string / reverted compared to the not truncated one hot encoding
def convert_fen_string_truncated(fen):
    output = np.zeros((12, 8, 8))

    split_fen = fen.split(" ")

    if (len(split_fen) != 1):
        print("Error: some fields in the FEN string are missing.")
        return

        ############################ Positions ############################
    # first dimension is for pieces. Begin with white pieces, pawn, knight, bishop, rook, queen, king. Example: blackte bishop = position 8
    # second dimension is the chess board row. So, from 1 to 8. (reverse ordering compared to fen string)
    # third dimension is the chess board column, so from a to h
    # position_converter = {'R': 0, 'N': 1, 'B': 2, 'Q': 3, 'K': 4, 'P': 5, 'r': 6, 'n': 7, 'b': 8, 'q': 9, 'k': 10, 'p': 11}

    index_row = 7
    index_column = 0
    for value in split_fen[0]:

        if (value == '/'):
            index_column = 0
            index_row -= 1
            continue

        if (value.isdigit()):
            index_column += int(value)
        else:
            index_piece = position_converter[value]
            output[index_piece, index_row, index_column] = 1
            index_column += 1


    # Dimensions: 12x8x8
    # Values: 12 for positions
    return output


# np.set_printoptions(threshold= sys.maxsize)
# print(convert_fen_string("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"))


def create_blank_emission_matrix(white=True):
    """
<<<<<<< HEAD
    Purpose of this is to create a blank emission matrix. Only information that it stores is the side of the board

    Current emission Encodding:
    Channels 1-12: stores result of sensing. emission_matrix[0, 2, 2] means there is a white rook at (2, 2)

    Channels 13-14: positions without piece types. Useful to know position of captured pieces

    Channel 15: positions of empty squares. Known from sensing and from moves

    Channel 16: black or white.


    :param white: are you white?
    :return: emission board with just white information written to it
    """
    emission_matrix = np.zeros((16, 8, 8))
    emission_matrix[-1, :, :] = int(white)
    return emission_matrix



def get_row_col_from_num(loc):
    """

    :param loc: board position as number
    :return: row, col of board
    """
    col = (loc) % 8 # it was loc-1 before
    row = loc // 8
    return row, col

#print(get_row_col_from_num(11))


def process_sense(sense_result, emission_matrix=np.zeros((16, 8, 8))):
    """
    Result of sensing

    Note: if you supply an emission matrix it is an in place operation
    :param sense_result: List of sense results
    :param emission_matrix: current emission matrix
    :return: modified emission matrix
    """
    for loc, piece in sense_result:
        if piece is not None:  # love how python is like english lol
            row, col = get_row_col_from_num(loc)

            piece_pos = position_converter[str(piece)]

            emission_matrix[piece_pos, row, col] = 1

        else: # empty square

            row, col = get_row_col_from_num(loc)
            emission_matrix[14, row, col] = 1

    return emission_matrix


def convert_channel(output, raw_truth_board, channel_truth, channel_truncated, expected_pieces):
    """
    This is a method that converts an output you have

    :param output: The truncated output representation we are filling in
    :param raw_truth_board: Florian's raw truth board
    :param channel: 0 - 12 int
    :param expected_pieces: How many pieces do we expect on that channel
    :return: void (output is updated)
    """
    # do White rooks (output channels 1 - 2)

    truthPieceChannel = raw_truth_board[channel_truth, :, :].flatten()
    piece_locations = np.argwhere(truthPieceChannel).flatten()
    pieces_alive = len(piece_locations)
    if pieces_alive > expected_pieces:
        print("A promotion probably occured")

    for i in range(min(pieces_alive, expected_pieces)):  # fill in known pieces
        output[channel_truncated + i, piece_locations[i]] = 1

    if pieces_alive < expected_pieces:  # fill in dead pieces (completes one hot encoding)
        output[channel_truncated + pieces_alive: channel_truncated + expected_pieces, -1] = 1

def get_truncated_board(truth_board):
    """
    Returns the truth board in the following format

    36 x 65.
    This is an alternate one hot encoding to the truth board. This one hot encoding is amenable to a softmax and allows
    us to to directly translate the predicted board into a valid hypothesis board.

    Channel breakdown [inclusive ordinal numbering]
    0-1   :R
    2-3   :N
    4-5   :B
    6-8   :Q  # cover the case where there are two queen promotions
    9     :K
    10-17 :P
    18-19 :r
    20-21 :n
    22-23 :B
    24-26 :q  # cover the case where there are two qeen promotions
    27    :k
    28-35 :p

    :param truth_board: gets truncated truth board from the truth board
    :return:
    """

    if isinstance(truth_board, type(np.array([]))):
        raw_truth_board = truth_board
    elif isinstance(truth_board, chess.Board):
        raw_truth_board = convert_fen_string(truth_board.fen())

    output = np.zeros((36, 65))  # 65 is for the graveuyard

    # WHITE PIECES
    convert_channel(output, raw_truth_board, 0, 0, 2)  # R at channel 0. There should be two of them
    convert_channel(output, raw_truth_board, 1, 2, 2)  # N at channel 1. There should be two of them
    convert_channel(output, raw_truth_board, 2, 4, 2)  # B at channel 2.
    convert_channel(output, raw_truth_board, 3, 6, 3)  # Q
    convert_channel(output, raw_truth_board, 4, 9, 1)  # K
    convert_channel(output, raw_truth_board, 5, 10, 8)  # P

    # BLACK PIECES
    convert_channel(output, raw_truth_board, 6, 18, 2)  # r
    convert_channel(output, raw_truth_board, 7, 20, 2)  # n
    convert_channel(output, raw_truth_board, 8, 22, 2)  # b
    convert_channel(output, raw_truth_board, 9, 24, 3)  # q
    convert_channel(output, raw_truth_board, 10, 27, 1)  # k
    convert_channel(output, raw_truth_board, 11, 28, 8)  # p

    # uncomment the assertion to get boast in run time
#    assert_truth_board_is_accurate(new_truth_board, raw_truth_board)
    return output


# np.set_printoptions(threshold= sys.maxsize)
# print(convert_fen_strinet("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"))

def start_bookkeeping(white=True):
    bookkeeping = np.zeros((12, 8, 8))
    if (white):
        pieces = [(0, 0, 0), (0, 0, 7), (1, 0, 1), (1, 0, 6), (2, 0, 2), (2, 0, 5), (3, 0, 3), (4, 0, 4), (5, 1, 0),
                  (5, 1, 1), (5, 1, 2), (5, 1, 3), (5, 1, 4), (5, 1, 5), (5, 1, 6), (5, 1, 7)]
    else:
        pieces = [(6, 7, 0), (6, 7, 7), (7, 7, 1), (7, 7, 6), (8, 7, 2), (8, 7, 5), (9, 7, 3), (10, 7, 4), (11, 6, 0),
                  (11, 6, 1), (11, 6, 2), (11, 6, 3), (11, 6, 4), (11, 6, 5), (11, 6, 6), (11, 6, 7)]

    for piece in pieces:
        bookkeeping[piece] = 1

    return bookkeeping

def get_truncated_board_short(truth_board, white = True):
    """
    Returns the truth board in the following format. If Florian completes his methods we can use this instead to train the RNN

    18 x 65.
    This is an alternate one hot encoding to the truth board. This one hot encoding is amenable to a softmax and allows
    us to to directly translate the predicted board into a valid hypothesis board.

    Channel breakdown [inclusive ordinal numbering]
    0-1   :r
    2-3   :n
    4-5   :b
    6-8   :q  # cover the case where there are two queen promotions
    9     :k
    10-17 :p


    :param truth_board: chess.Board object
    :param white: Whether or not we are extracting just white, or just black. (If you are training on white then you
    want white to be False) :param truth_board: gets truncated truth board from the truth board :return:
    """
    if isinstance(truth_board,  type(np.array([]))):
        raw_truth_board = truth_board
    elif isinstance(truth_board, chess.Board):
        raw_truth_board = convert_fen_string(truth_board.fen())

    output = np.zeros((18, 65))  # 65 is for the graveyard

    # Need to predict black pieces
    if not white:
        convert_channel(output, raw_truth_board, 6, 0, 2)  # R at channel 0. There should be two of them
        convert_channel(output, raw_truth_board, 7, 2, 2)  # N at channel 1. There should be two of them
        convert_channel(output, raw_truth_board, 8, 4, 2)  # B at channel 2.
        convert_channel(output, raw_truth_board, 9, 6, 3)  # Q
        convert_channel(output, raw_truth_board, 10, 9, 1)  # K
        convert_channel(output, raw_truth_board, 11, 10, 8)  # P

        output_padded = np.zeros((36, 65))
        output_padded[18:] = output
        new_truth_board = convert_truncated_to_truth(output_padded)
        assert_truth_board_is_accurate(new_truth_board[6:], raw_truth_board[6:])

    else:
        convert_channel(output, raw_truth_board, 0, 0, 2)  # r
        convert_channel(output, raw_truth_board, 1, 2, 2)  # n
        convert_channel(output, raw_truth_board, 2, 4, 2)  # b
        convert_channel(output, raw_truth_board, 3, 6, 3)  # q
        convert_channel(output, raw_truth_board, 4, 9, 1)  # k
        convert_channel(output, raw_truth_board, 5, 10, 8)  # p

        output_padded = np.zeros((36, 65))
        output_padded[:18] = output
        new_truth_board = convert_truncated_to_truth(output_padded)
        assert_truth_board_is_accurate(new_truth_board[:6], raw_truth_board[:6])
    return output




def fill_channel(truncated_board, raw_truth_board, piece_truncated_index, piece_channel_index, expected_pieces):
    """
    This is the opposite of convert channel. It fills in the truth board from the truncated board

    :param output: The truncated output representation we are filling in
    :param raw_truth_board: Florian's raw truth board
    :param channel: 0 - 12 int
    :param expected_pieces: How many pieces do we expect on that channel
    :return: void (output is updated)
    """

    sum_truncated_rooks = np.sum(truncated_board[piece_truncated_index:piece_truncated_index + expected_pieces, :64],
                                 axis=0)

    raw_truth_board[piece_channel_index] = sum_truncated_rooks.reshape(8, 8)


def convert_truncated_to_truth(truncated_board):
    """
    Opposite of get_truncated_board. Important for testing truncated and also recreating truth board from logits
    :param truncated_board:
    :return:
    """

    raw_truth_board = np.zeros((12, 8, 8))

    # WHITE
    fill_channel(truncated_board, raw_truth_board, 0, 0, 2)  # R
    fill_channel(truncated_board, raw_truth_board, 2, 1, 2)  # N
    fill_channel(truncated_board, raw_truth_board, 4, 2, 2)  # B
    fill_channel(truncated_board, raw_truth_board, 6, 3, 3)  # Q
    fill_channel(truncated_board, raw_truth_board, 9, 4, 1)  # K
    fill_channel(truncated_board, raw_truth_board, 10, 5, 8)  # P

    # BLACK
    fill_channel(truncated_board, raw_truth_board, 18, 6, 2)  # r
    fill_channel(truncated_board, raw_truth_board, 20, 7, 2)  # n
    fill_channel(truncated_board, raw_truth_board, 22, 8, 2)  # b
    fill_channel(truncated_board, raw_truth_board, 24, 9, 3)  # q
    fill_channel(truncated_board, raw_truth_board, 27, 10, 1)  # k
    fill_channel(truncated_board, raw_truth_board, 28, 11, 8)  # p

    return raw_truth_board


def assert_truth_board_is_accurate(new_truth_board, truth_board):
    """
     Compares reconstructed truth board from truth board. Error checking

     :param new_truth_board: reconstructed truth board
     :param truth_board:
     :return:
     """
    for i in range(new_truth_board.shape[0]):
        channel_new = new_truth_board[i]
        channel_truth = truth_board[i][:12]

        truth_args = np.argwhere(channel_truth)
        new_args = np.argwhere(channel_new)

        for arg in new_args:
            assert arg in truth_args  # can underestimate in the case of promotions

        assert len(truth_args) >= len(new_args)

def find_piece_type(bookkeeping, row: int, column: int):
    """
    It looks at the bookkeeping, the row and the column, and it tells you which piece is on that square

    :param bookkeeping:
    :param row:
    :param column:
    :return:
    """
    for i in range (12):
        if (bookkeeping[i,row,column] == 1):
            return i
        
    raise Exception("Sorry, that square does not have any piece on it")  #book

def get_most_likely_truth_board(truncated_board: torch.Tensor, emission_matrix: np.array, white):
    """

    :param truncated_board: Raw output of neural network
    :param emission_matrix: emission matrix filled with true facts
    :return:
    """
    if len(truncated_board.shape) == 3:
        truncated_board = truncated_board[0]
    elif len(truncated_board.shape) != 2:
        print(truncated_board.shape)
        raise(Exception("Truncated board is wrong shape"))

    known_pieces = emission_matrix[:12]
    softmax_known = get_truncated_board(known_pieces)
    if white:
        softmax_known[18:, -1] = 0  # only remove dead pieces for black
    else:
        softmax_known[:18, -1] = 0  # remove only dead white

    softmax_unknown = torch.nn.functional.softmax(truncated_board,
                                                  dim=1).detach().cpu().numpy()  # softmax is over black
    if white:
        softmax = np.vstack((np.zeros((18, 65)), softmax_unknown))  #only predicting blacks pieces
    else:
        softmax = np.vstack((softmax_unknown, np.zeros((18, 65))))  # only predicting white's pieces

    max_truncated_board = np.zeros(softmax.shape) # board with maximum probability. Use greedy heuristic
    # fill in the known pieces first.
    for channel_idx in range(softmax_known.shape[0]):
        if np.any(softmax_known[channel_idx]):
            max_piece_pos = np.argmax(softmax_known[channel_idx])

            max_truncated_board[channel_idx, max_piece_pos] = 1
            softmax[channel_idx, :] = -1  # cover up this row so we don't pick it again

            if max_piece_pos != 64:
                softmax[:, max_piece_pos] = -1  # do not pick any pieces in this spot

    assert np.all(convert_truncated_to_truth(max_truncated_board) == known_pieces)  # have you filled in the known pieces correctly

    # loop greedily finds the most likely piece one at a time and prevents other pieces from being on that square
    while True:
        max_probs = softmax.max(axis=1)
        max_piece = np.argmax(max_probs)
        if max_probs[max_piece] == -1:
            break

        max_piece_pos = np.argmax(softmax[max_piece])
        max_truncated_board[max_piece, max_piece_pos] = 1

        softmax[max_piece, :] = -1  # cover up this row so we don't pick it again
        if max_piece_pos != 64:  #
            softmax[:, max_piece_pos] = -1  # do not pick any pieces in this spot



    return convert_truncated_to_truth(max_truncated_board)

def convert_one_hot_to_board(one_hot_board):
    """

    :param one_hot_board:
    :return:
    """
    board = chess.Board()
    piece_map = board.piece_map()

    string_to_piece_map = {} # maps string repr of pieces to chess.Piece objects
    for key, val in piece_map.items():
        string_to_piece_map[str(val)] = val

    board.clear_board() # clear board

    piece_str_dict = {v: k for k, v in position_converter.items()}  # inverse if position convert

    for channel_idx in range(one_hot_board.shape[0]):
        channel = one_hot_board[channel_idx]
        piece_str = piece_str_dict[channel_idx]
        squares = np.argwhere(channel)
        for row, col in squares:

            # col = (loc - 1) % 8
            # row = loc // 8
            piece_loc = (row) * 8 + col
            board.set_piece_at(piece_loc, string_to_piece_map[piece_str])

    board_one_hot_converted = convert_fen_string(board.fen())

    #assert np.all(board_one_hot_converted[:12, :, :] == one_hot_board)
    return board

def assert_bookkeeping_is_accurate(bookkeeping, board, white):
    fen = board.board_fen()
    matrix = convert_fen_string_truncated(fen)
    if white: 
        if np.array_equal(matrix[:6], bookkeeping[:6]):
            return True
    else:
        if np.array_equal(matrix[6:12], bookkeeping[6:12]):
            return True
        
    return False

def piece_type_converter(piece_type, white):
    dic = {1: 5, 2: 1, 3: 2, 4: 0, 5: 3, 6: 4}
    output = dic[piece_type]
    if white:
        return output
    else:
        return (output + 6)
    
def is_knight_rush(sense_result, color):
    '''
    :param sense_result: A list of tuples, where each tuple contains a :class:`Square` in the sense, and if there
                             was a piece on the square, then the corresponding :class:`chess.Piece`, otherwise `None`.
                             
    :param color: color of the player that is playing. We don't care about our own pieces
    '''
    #reminder: pawn 1, knight 2, bishop 3, rook 4, queen 5, king 6
    dic_white = {1: ['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2'], 2: ['b1', 'g1'], 3: ['c1', 'f1'], 4: ['a1', 'h1'], 5: ['d1'], 6: ['e1']}
    dic_black = {1: ['a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7'], 2: ['b8', 'g8'], 3: ['c8', 'f8'], 4: ['a8', 'h8'], 5: ['d8'], 6: ['e8']}
    
    for sense in sense_result:
        if (sense[1] != None):
            #Here, we sensed a piece
            if (sense[1].color != color):
                #a piece that belongs to our opponent
                square_name = chess.square_name(sense[0])
                piece_type = sense[1].piece_type
                
                if (piece_type == 2):
                    continue
                
                #an opponent piece that is not a knight
                if (color == True):
                    if (square_name in dic_black[piece_type]):
                        continue
                    else:
                        return False
                else:
                    if (square_name in dic_white[piece_type]):
                        continue
                    else:
                        return False
                    
    return True
                
                
            
    
    