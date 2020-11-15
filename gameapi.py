#!/usr/bin/env python

import chess
import hyperparams
import numpy
from fen_string_convert import convert_fen_string
from math import floor

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
        self.clock = 0

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
        board = convert_fen_string(self.board_fen)
        return board

    def getValidMoves(self):
        """ Return list of valid pseudo legal moves.  """
        # valid_moves is a list of chess.Move objects
        valid_moves = self.board.generate_pseudo_legal_moves()
        # action_mask is a 1-hot encoded vector of size 4673 version of valid_moves
        action_mask = [0] * self.getActionSize()


        # For every move in valid_moves, fill a 1 into the appropriate element
        # of action_mask
        action_mask[-1] = 1 # for pass action
        
        column_convert = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7}
        
        for move in valid_moves:
            # get squares from the move
            from_square = move.from_square
            to_square = move.to_square
            
            # get names of the squares
            from_square_name = chess.square_name(from_square)
            to_square_name = chess.square_name(to_square)
            
            # get numerical position of squares
            from_square_position = [column_convert(from_square_name[0]), int(from_square_name[1])]
            to_square_position = [column_convert(to_square_name[0]), int(to_square_name[1])]
            
            # get the variation in rows and columns characterizing the move
            column_variation = to_square_position[0] - from_square_position[0]
            row_variation = to_square_position[1] - from_square_position[0]
            variation = (column_variation, row_variation)
            
            def get_move_index(i):
                return (73 *(8 * (from_square_position[0] - 1) + from_square_position[1] - 1 ) + i)
            
            if (move.promotion != None):
                uci = move.uci()
                promotion_type = uci[-1]
                if (promotion_type == 'r'):
                    action_mask[get_move_index(65 + column_variation)] = 1
                    continue
                elif (promotion_type == 'b'):
                    action_mask[get_move_index(68 + column_variation)] = 1
                    continue
                elif (promotion_type == 'n'):
                    action_mask[get_move_index(71 + column_variation)] = 1
                    continue
                #else, it's a queen promotion
                
                
            # bishop move
            if (abs(column_variation) == abs(row_variation)):
                
                if (column_variation >= 0):
                    if (row_variation >= 0):
                        action_mask[get_move_index(35 + column_variation - 1)] = 1
                    else:
                        action_mask[get_move_index(42 + column_variation - 1)] = 1
                else:
                    if (row_variation >= 0):
                        action_mask[get_move_index(28 - column_variation - 1)] = 1
                    else:
                        action_mask[get_move_index(49 - column_variation - 1)] = 1
            
            #rook move
            elif (column_variation == 0):
                if (row_variation >= 0):
                    action_mask[get_move_index(7 + row_variation - 1)] = 1
                else:
                    action_mask[get_move_index(21 - row_variation - 1)] = 1
            
            #another rook move
            elif (row_variation == 0):
                if (column_variation >= 0):
                    action_mask[get_move_index(14 + column_variation - 1)] = 1
                else:
                    action_mask[get_move_index(0 - column_variation - 1)] = 1
                 
            #knight move
            else:
                knight_moves = {(-2,-1): 56, (-2,1): 57, (-1,2): 58, (1,2): 59, (2,1): 60, (2,-1): 61, (1,-2): 62, (-1,-2): 63}
            
                action_mask[get_move_index(knight_moves(variation))] = 1
        

        return action_mask


    def make_move(self, action):
        """ Make the move """
        # TO-DO: implement normal promotion from a pawn to a queen. Currently only underpromotions are supported.
        # action is a 1-hot encoded vector of size 4673.  Convert it to
        # a chess.Move object
        index = action.index(action)
        action_type = index % 73
        row = int(((index - action_type) % (73*8)) / 73) #starts at 0
        column = floor(index / (73*8)) # starts at 0
    
        
        if (action_type >= 64 and action_type <= 66):
            if (row == 6):
                move = chess.Move(chess.square(column, row), chess.square(column - 65 + action_type, row+1),chess.ROOK)
            else:
                move = chess.Move(chess.square(column, row), chess.square(column - 65 + action_type, row - 1),chess.ROOK)
                
        elif (action_type >= 67 and action_type <= 69):
            if (row == 6):
                move = chess.Move(chess.square(column, row), chess.square(column - 68 + action_type, row+1),chess.BISHOP)
            else:
                move = chess.Move(chess.square(column, row), chess.square(column - 68 + action_type, row - 1),chess.BISHOP)
                
        elif (action_type >= 70 and action_type <= 72):
            if (row == 6):
                move = chess.Move(chess.square(column, row), chess.square(column - 71 + action_type, row+1),chess.KNIGHT)
            else:
                move = chess.Move(chess.square(column, row), chess.square(column - 71 + action_type, row - 1),chess.KNIGHT)
        
        elif (action_type == 73):
            move = chess.Move.null()
    
        elif (action_type <= 6):
            dest_row = row
            dest_col = column - action_type
            move = chess.Move(chess.square(column, row), chess.square(dest_col, dest_row))
        elif (action_type >= 7 and action_type <= 13):
            dest_col = column
            dest_row = row + action_type - 6
            move = chess.Move(chess.square(column, row), chess.square(dest_col, dest_row))
        elif (action_type >= 14 and action_type <= 20):
            dest_row = row
            dest_col = column + action_type - 13
            move = chess.Move(chess.square(column, row), chess.square(dest_col, dest_row))
        elif (action_type >= 21 and action_type <= 27):
            dest_col = column
            dest_row = row - action_type + 20
            move = chess.Move(chess.square(column, row), chess.square(dest_col, dest_row))
        elif (action_type >= 28 and action_type <= 34):
            dest_col = column - action_type + 27
            dest_row = row + action_type - 27
            move = chess.Move(chess.square(column, row), chess.square(dest_col, dest_row))
        elif (action_type >= 35 and action_type <= 41):
            dest_col = column + action_type - 34
            dest_row = row + action_type - 34
            move = chess.Move(chess.square(column, row), chess.square(dest_col, dest_row))
        elif (action_type >= 42 and action_type <= 48):
            dest_col = column + action_type - 41
            dest_row = row - action_type + 41
            move = chess.Move(chess.square(column, row), chess.square(dest_col, dest_row))
        elif (action_type >= 49 and action_type <= 55):
            dest_col = column - action_type + 48
            dest_row = row - action_type + 48
            move = chess.Move(chess.square(column, row), chess.square(dest_col, dest_row))
        elif (action_type >= 56 and action_type <= 63):
            knight_moves = {56: (-2,-1), 57: (-2,1), 58: (-1,2), 59: (2,1), 60: (2,1), 61: (2,-1), 62: (1,-2), 63: (-1,-2)}
            knight_move = knight_moves(action_type)
            dest_row = knight_move[1] + row
            dest_col = knight_move[0] + column
            move = chess.Move(chess.square(column, row), chess.square(dest_col, dest_row))
            

        self.board.push(move)
        return


    def time_left(self):
        """ Return the time left in seconds """
        # Not needed for training since we'll use a hardcoded limit for number
        # of MCTS simulations, but when playing a tournament game, we need to keep track
        # of how much time we have left and stop MCTS simulations when time is almost up
        pass
    
