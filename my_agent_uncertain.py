#!/usr/bin/env python3

"""
File Name:      my_agent.py
Authors:        TODO: Your names here!
Date:           TODO: The date you finally started working on this.

Description:    Python file for my agent.
Source:         Adapted from recon-chess (https://pypi.org/project/reconchess/)
"""

import random
import chess
from player import Player
from mcts import MCTS
from rbmcnet import RbmcNet, NNetWrapper
from gameapi import GameAPI
import torch
import chess.engine
import numpy as np
from stockfish import Stockfish
from uncertainty_rnn import BoardGuesserNetOnline
from fen_string_convert import process_sense, create_blank_emission_matrix, get_row_col_from_num, get_most_likely_truth_board, convert_one_hot_to_board


# TODO: Rename this class to what you would like your bot to be named during the game.
class AnotherChessBot(Player):

    def __init__(self):

        self.color = None
        self.board = None

        self.load_weights = True

    def handle_game_start(self, color, board):
        """
        This function is called at the start of the game.

        :param color: chess.BLACK or chess.WHITE -- your color assignment for the game
        :param board: chess.Board -- initial board state
        :return:
        """
        # TODO: implement this method
        self.board = board
        self.color = color
        self.move_count = 0
        self.use_stockfish = False

        self.network = BoardGuesserNetOnline() # neural network for inferring truth board
        if color:
            self.network.load_state_dict(torch.load("white_rnn_model"))
        else:
            self.network.load_state_dict(torch.load("black_rnn_model"))

        self.white = color
        self.hidden = None
        self.emission_matrix = create_blank_emission_matrix(self.white) # Sensing Matrix. Florian modify this!
        self.nnet = NNetWrapper()

        if self.load_weights:
            self.nnet.load_checkpoint(folder="models", filename="stockfish.pth.tar")

    def handle_opponent_move_result(self, captured_piece, captured_square):
        """
        This function is called at the start of your turn and gives you the chance to update your board.

        :param captured_piece: bool - true if your opponents captured your piece with their last move
        :param captured_square: chess.Square - position where your piece was captured
        """

        if captured_piece:
            row, col = get_row_col_from_num(captured_square)
            self.emission_matrix[12, row, col] = 1
        pass

    def choose_sense(self, possible_sense, possible_moves, seconds_left):
        """
        This function is called to choose a square to perform a sense on.

        :param possible_sense: List(chess.SQUARES) -- list of squares to sense around
        :param possible_moves: List(chess.Moves) -- list of acceptable moves based on current board
        :param seconds_left: float -- seconds left in the game

        :return: chess.SQUARE -- the center of 3x3 section of the board you want to sense
        :example: choice = chess.A1
        """
        # TODO: update this method
        return random.choice(possible_sense)

    def handle_sense_result(self, sense_result):
        """
        This is a function called after your picked your 3x3 square to sense and gives you the chance to update your
        board.

        :param sense_result: A list of tuples, where each tuple contains a :class:`Square` in the sense, and if there
                             was a piece on the square, then the corresponding :class:`chess.Piece`, otherwise `None`.
        :example:
        [
            (A8, Piece(ROOK, BLACK)), (B8, Piece(KNIGHT, BLACK)), (C8, Piece(BISHOP, BLACK)),
            (A7, Piece(PAWN, BLACK)), (B7, Piece(PAWN, BLACK)), (C7, Piece(PAWN, BLACK)),
            (A6, None), (B6, None), (C8, None)
        ]
        """
        # TODO: implement this method
        # Hint: until this method is implemented, any senses you make will be lost.
        process_sense(sense_result, self.emission_matrix)  # adds sensing information to emission matrix

        pass

    def choose_move(self, possible_moves, seconds_left):
        """
        Choose a move to enact from a list of possible moves.

        :param possible_moves: List(chess.Moves) -- list of acceptable moves based only on pieces
        :param seconds_left: float -- seconds left to make a move

        :return: chess.Move -- object that includes the square you're moving from to the square you're moving to
        :example: choice = chess.Move(chess.F2, chess.F4)

        :condition: If you intend to move a pawn for promotion other than Queen, please specify the promotion parameter
        :example: choice = chess.Move(chess.G7, chess.G8, promotion=chess.KNIGHT) *default is Queen
        """
        # TODO: update this method

        # Use rnn to figure out state
        self.softmax_out, self.hidden = self.network(torch.Tensor([self.emission_matrix]), self.hidden)
        pred_board = get_most_likely_truth_board(self.softmax_out)

        current_board = convert_one_hot_to_board(pred_board)
        current_board.turn = self.color
        self.move_count += 1
        # whom: Run MCTS simulation for num_simulations and then choose the best move
        # Provide current_board (or what we believe to be the current board) to MCTS
        # During training self.board has been modified to be the truth board

        #current_board = self.board

        gameapi = GameAPI(current_board)
        gameapi.print_board()

        # Endgame move to capture the opponent king
        move = gameapi.end_game_move(self.color)
        if move is not None:
            return move

        mcts = MCTS(gameapi, self.nnet, num_mcts_sims=2, cpuct=1.0)

        probs = mcts.getActionProb()
        best_move = np.argmax(probs)

        # convert best_move to Move object
        move = gameapi.make_move(best_move, apply_move=False)

        if self.use_stockfish:
            stockfish = Stockfish("/usr/games/stockfish")
            stockfish.set_fen_position(gameapi.fen)

            move = stockfish.get_best_move()
            move = chess.Move.from_uci(move)

        self.emission_matrix = create_blank_emission_matrix(self.white)  # clear it here
        return move

    def handle_move_result(self, requested_move, taken_move, reason, captured_piece, captured_square):
        """
        This is a function called at the end of your turn/after your move was made and gives you the chance to update
        your board.

        :param requested_move: chess.Move -- the move you intended to make
        :param taken_move: chess.Move -- the move that was actually made
        :param reason: String -- description of the result from trying to make requested_move
        :param captured_piece: bool - true if you captured your opponents piece
        :param captured_square: chess.Square - position where you captured the piece
        """
        # TODO: implement this method
        if requested_move != None:
            from_row, from_col = get_row_col_from_num(requested_move.from_square)
            self.emission_matrix[13, from_row, from_col] = 1

            to_row, to_col = get_row_col_from_num(requested_move.to_square)
            self.emission_matrix[14, from_row, from_col] = 1

        if taken_move != None:  # what was the move you actually took
            from_row, from_col = get_row_col_from_num(taken_move.from_square)
            self.emission_matrix[15, from_row, from_col] = 1

            to_row, to_col = get_row_col_from_num(taken_move.to_square)
            self.emission_matrix[16, from_row, from_col] = 1

        if captured_piece:  # did you capture a piece
            self.emission_matrix[17,:, :] = 1

        pass

    def handle_game_end(self, winner_color, win_reason):  # possible GameHistory object...
        """
        This function is called at the end of the game to declare a winner.

        :param winner_color: Chess.BLACK/chess.WHITE -- the winning color
        :param win_reason: String -- the reason for the game ending
        """
        # TODO: implement this method
        pass


def format_print_board(board):
    rows = ['8', '7', '6', '5', '4', '3', '2', '1']
    fen = board.board_fen()

    fb = "   A   B   C   D   E   F   G   H  "
    fb += rows[0]
    ind = 1
    for f in fen:
        if f == '/':
            fb += '|' + rows[ind]
            ind += 1
        elif f.isnumeric():
            for i in range(int(f)):
                fb += '|   '
        else:
            fb += '| ' + f + ' '
    fb += '|'

    ind = 0
    for i in range(9):
        for j in range(34):
            print(fb[ind], end='')
            ind += 1
        print('\n', end='')
    print("")

