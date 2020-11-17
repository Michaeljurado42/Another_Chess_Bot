#!/usr/bin/env python3

"""
File Name:      random_agent.py
Authors:        Michael Johnson and Leng Ghuy
Date:           March 9th, 2019

Description:    Python file of a random bot
Source:         Adapted from recon-chess (https://pypi.org/project/reconchess/)
"""

import random
import chess
from player import Player

from fen_string_convert import process_sense, convert_fen_string, get_row_col_from_num, create_blank_emission_matrix

import numpy as np
class Random(Player):

    def handle_game_start(self, color, white):
        """
        This function is called at the start of the game.

        :param color: chess.BLACK or chess.WHITE -- your color assignment for the game
        :param board: chess.Board -- initial board state
        """
        self.white = white
        self.emission_matrix = create_blank_emission_matrix(self.white)

    def handle_opponent_move_result(self, captured_piece, captured_square):
        """
        This function is called at the start of your turn and gives you the chance to update your board.

        :param captured_piece: bool - true if your opponents captured your piece with their last move
        :param captured_square: chess.Square - position where your piece was captured
        """

        if captured_piece:
            row, col = get_row_col_from_num(captured_square)
            self.emission_matrix[12, row, col] = 1


    def choose_sense(self, possible_sense, possible_moves, seconds_left):
        """
        This function is called to choose a square to perform a sense on.

        :param possible_sense: List(chess.SQUARES) -- list of squares to sense around
        :param possible_moves: List(chess.Moves) -- list of acceptable moves based on current board
        :param seconds_left: float -- seconds left in the game

        :return: chess.SQUARE -- the center of 3x3 section of the board you want to sense
        :example: choice = chess.A1
        """

        # use emsission matrix here for inference
        # TODO

        # neural network stuff
        self.emission_matrix = create_blank_emission_matrix(self.white)  # only clear when you have used the matrix as input to RNN
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

        # Use rnn to figure out state

        # use mcts and policy network


        self.emission_matrix = create_blank_emission_matrix(self.white)  # clear it here

        return random.choice(possible_moves)

    def handle_move_result(self, requested_move, taken_move, reason, captured_piece, captured_square):
        """
        This is a function called at the end of your turn/after yourg move was made and gives you the chance to update
        your board.

        :param requested_move: chess.Move -- the move you intended to make
        :param taken_move: chess.Move -- the move that was actually made
        :param reason: String -- description of the result from trying to make requested_move
        :param captured_piece: bool -- true if you captured your opponents piece
        :param captured_square: chess.Square -- position where you captured the piece
        """

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



    def handle_game_end(self, winner_color, win_reason):  # possible GameHistory object...
        """
        This function is called at the end of the game to declare a winner.

        :param winner_color: Chess.BLACK/chess.WHITE -- the winning color
        :param win_reason: String -- the reason for the game ending
        """
        pass
