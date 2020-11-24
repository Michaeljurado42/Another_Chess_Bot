#!/bin/bash

NUM_AGENTS=4
NUM_GAMES=25

# Run test1
python test1.py

echo "Playing against random bot with my_agent as White Player"
python play_many_games.py my_agent.py random_agent.py --quiet True --num_agents $NUM_AGENTS --num_games $NUM_GAMES

echo "Playing against random bot with my_agent as Black Player"
python play_many_games.py random_agent.py my_agent.py --quiet True --num_agents $NUM_AGENTS --num_games $NUM_GAMES

echo "Playing against Knight rush bot with my_agent as White Player"
python play_many_games.py my_agent.py knight_rush.py --quiet True --num_agents $NUM_AGENTS --num_games $NUM_GAMES

echo "Playing against Knight rush bot with my_agent as Black Player"
python play_many_games.py knight_rush.py my_agent.py --quiet True --num_agents $NUM_AGENTS --num_games $NUM_GAMES
