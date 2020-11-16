#%%
import numpy as np
import chess
from game import *
from play_game import *
from gameapi import GameAPI

#%%

game = Game()

board = game.truth_board
#
#%%

def play_random_game():
    name_one, constructor_one = load_player("random_agent.py")
    name_two, constructor_two = load_player("random_agent.py")
    white_player, black_player = constructor_one(), constructor_two()

    player_names = [name_one, name_two]

    game = Game()
    white_player.handle_game_start(chess.WHITE, game.truth_board)
    black_player.handle_game_start(chess.BLACK, game.truth_board)

    game.start()
    move_number = 0
    while not game.is_over():
        move_number += 1
        player = white_player if game.turn else black_player
        possible_sense = list(chess.SQUARES)

        possible_moves = game.get_moves()

        board = game.truth_board

        test_moves(board)

        # notify the player of the previous opponent's move
        captured_square = game.opponent_move_result()
        player.handle_opponent_move_result(captured_square is not None, captured_square)

        # play sense action
        sense = player.choose_sense(possible_sense, possible_moves, game.get_seconds_left())
        sense_result = game.handle_sense(sense)
        player.handle_sense_result(sense_result)

        # play move action
        move = player.choose_move(possible_moves, game.get_seconds_left())
        requested_move, taken_move, captured_square, reason = game.handle_move(move)
        player.handle_move_result(requested_move, taken_move, reason, captured_square is not None,
                              captured_square)


        game.end_turn()
    print("Finished testing a game with", move_number, "moves")
    print("Test passed!")


def test_moves(board):
    """ For every possible move from this board state, run it through getValidMoves to get
        the actions vector (size 4673) representation.  Then run this back through make_move
        and compare to see if we get the same move uci string 
    """

    mcts_game = GameAPI(board)
    #mcts_game.print_board()
    moves = list(board.generate_legal_moves())

    for move in moves:
        actions = mcts_game.getValidMoves(moves=[move])
        actions[-1] = 0
        the_move = mcts_game.make_move(actions)

        move_str = move.uci()
        if len(move_str) == 5 and move_str[4] == 'q':
            # Our representation doesn't tag the 'q'
            move_str = move_str[:-1]
            
        ## Ensure that we are getting back the same move string ##
        if move_str != the_move.uci():
            print(move_str, "!=", the_move.uci())
            print(mcts_game.stringRepresentation())
            assert(False)

        mcts_game.pop()

#%%
play_random_game()
