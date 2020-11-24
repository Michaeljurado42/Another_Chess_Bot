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
    #print(board.fen())
    moves = list(board.generate_legal_moves())

    for move in moves:
        actions = mcts_game.getValidMoves(moves=[move])
        action = np.nonzero(actions)[0][0]
        the_move = mcts_game.make_move(action)

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



#%%
# Test pawn capture for black
fen = "K7/8/8/8/8/8/p7/RN6 b KQkq - 0 1"
board.set_fen(fen)

#format_print_board(board)

moves = list(board.generate_legal_moves())

test_moves(board)

print("Test pawn capture for black passed")

# %%

# Test pawn capture for white
fen = "Kq6/P7/8/8/8/8/p7/R7 w KQkq - 0 1"
board.set_fen(fen)

#format_print_board(board)

moves = list(board.generate_legal_moves())

test_moves(board)

print("Test pawn capture for white passed")



# %%

# Test pass
fen = "r3k1nr/p2p2pp/n1p1p3/1p2Np2/1bP1Bq2/2N5/PP1P1PPP/R1BQK2R b KQkq - 1 16"

board.set_fen(fen)
assert(board.fen() == fen)

gameapi = GameAPI(board)
assert(gameapi.board.fen() == fen)

# The null move is where the last element is a 1
action = gameapi.getActionSize() - 1

move = gameapi.make_move(action)

# A null move is represented as "0000"
assert(move.uci() == "0000")

# Turn should have changed
assert(gameapi.board.fen() != fen)
# ... board pieces should not have moved
assert(gameapi.board.board_fen() == board.board_fen())

print("Null move test passed")


# %%
