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
    move_number = 1
    while not game.is_over():
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



def test_moves(board):
    """ For every possible move from this board state, run it through getValidMoves to get
        the actions vector (size 4673) representation.  Then run this back through make_move
        and compare to see if we get the same move uci string 
    """

    mcts_game = GameAPI(board)
    mcts_game.print_board()
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
            assert(False)

        mcts_game.pop()

#%%
play_random_game()

#%%

fen1 = '7k/1P6/8/8/8/8/8/8 w - - 0 74'
prepare_test(board, fen1)



# %%
 %%
fen1 = '7k/8/8/8/8/8/8/K7 w - - 0 74'

board.set_fen(fen1)

initial_state = board._board_state

# %%

format_print_board(board)
# %%

# Print list of legal moves
# not sure what's the difference between this api and 
# board.generate_pseudo_legal_moves()
moves =list(board.generate_legal_moves())
for move in moves:
    print(move.uci(), end=' ')

len(moves)


# %%
mcts_game = GameAPI(board)

# %%

mcts_game.stringRepresentation()
# %%
valid_moves = mcts_game.getValidMoves()

#%%

np.nonzero(np.array(valid_moves))

# should be equal to [7, 14, 35, 4672]

# %%

# Test 2
fen1 = '7k/8/8/8/8/8/6K1/8 w - - 0 74'

board.set_fen(fen1)
format_print_board(board)
moves =list(board.generate_legal_moves())
for move in moves:
    print(move.uci(), end=' ')

len(moves)


mcts_game = GameAPI(board)
mcts_game.stringRepresentation()



# %%
valid_moves = mcts_game.getValidMoves()
values = np.nonzero(np.array(valid_moves))
## should get set   ([3619, 3591, 3612, 3598, 3584, 3626, 3577, 3605])
values


# %%
# Test 3
fen1 = '7k/1P6/8/8/8/8/8/8 w - - 0 74'

board.set_fen(fen1)
format_print_board(board)
moves =list(board.generate_legal_moves())
for move in moves:
    print(move.uci(), end=' ')

len(moves)


mcts_game = GameAPI(board)
mcts_game.stringRepresentation()


# %%
valid_moves = mcts_game.getValidMoves()
values = np.nonzero(np.array(valid_moves))

values
# %%

# tests for make_move
actions = [0] * 4673
actions[1029] = 1
actions = np.array(actions)

fen1 = '7k/1P6/8/8/8/8/8/8 w - - 0 74'

board.set_fen(fen1)
format_print_board(board)
moves =list(board.generate_legal_moves())
for move in moves:
    print(move.uci(), end=' ')

len(moves)

mcts_game = GameAPI(board)
mcts_game.stringRepresentation()
move = mcts_game.make_move(actions)
mcts_game.print_board()
move

actions2 = mcts_game.getValidMoves([move])

actions[4672] = 1

assert(np.sum(actions == actions2) == 4673)



#%%
np.nonzero(actions)

# test submit

# %%
# %%
