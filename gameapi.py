#!/usr/bin/env python

import chess
import hyperparams
import numpy

class GameAPI():
    """ Our GameAPI class which will be passed into MCTS.  This class will help
    MCTS code figure out when a game is over, what the legal moves are, convert
    board state -> fen string etc. 
    """

    def __init__(self, board, time_limit=None):
        # Store the current state of the board at the beginning of our players turn
        # self.board_fen can then be used to restore to this state every time we do
        # a new simulation in MCTS
        self.board_fen = board.board_fen()
        self.board = chess.Board()
        self.board.set_fen(self.board_fen)

        # Probably should keep track of time-limit ourselves unless we
        # have access to Game object from the template code
        self.clock = ...

    def restore_board(self):
        """ Restore back to the board_fen when this GameAPI was constructed.  Can be called at the start of 
        a new simulation to change board state back to root node """
        self.board.set_fen(self.board_fen)

    def stringRepresentation(self):
        return self.board.board_fen()


    def getCanonicalBoardSize(self):
        """ Return the dimensions which will be input to the neural network (channels, board_x, board_y) """
        return hyperparams.input_dims

    def getActionSize(self):
        """ Return action size """
        return hyperparams.action_size

    def getGameEnded(self):
        """ Game is ended when one of the kings are missing """
        king_captured = self.board.king(chess.WHITE) is None or self.board.king(chess.BLACK) is None
        return king_captured

    def getCanonicalBoard(self):
        """ Returns the 20x8x8 board that can be fed to our neural network """
        # TO-DO:
        # Maybe make use of the convert_fen_string function here
        board = numpy.zeros((hyperparams.input_dims))
        return board

    def getValidMoves(self):
        """ Return list of valid pseudo legal moves.  """
        # valid_moves is a list of chess.Move objects
        valid_moves = self.board.generate_pseudo_legal_moves()
        # action_mask is a 1-hot encoded vector of size 4673 version of valid_moves
        action_mask = [0] * self.getActionSize()

        # TO-DO:
        # For every move in valid_moves, need to fill a 1 into the appropriate element
        # of action_mask
        ...

        return action_mask


    def make_move(self, action):
        """ Make the move """
        # TO-DO:
        # action is a 1-hot encoded vector of size 4673.  Need to convert it to
        # a chess.Move object
        move = ...

        self.board.push(move)
        return


    def time_left(self):
        """ Return the time left in seconds """
        # Not needed for training since we'll use a hardcoded limit for number
        # of MCTS simulations, but when playing a tournament game, we need to keep track
        # of how much time we have left and stop MCTS simulations when time is almost up
        pass
    
