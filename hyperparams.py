#!/usr/bin/env python

# Run with GPU.  Set this to false if 
# getting runtime errors on version python 3.5
# Seems to run fine for python 3.7.9
use_gpu = False

# Variables for training neural network
batch_size = 128
epochs = 20

# Input dimensions of canonical board that is fed into neural network
input_dims = (20, 8, 8)

# For the action probabilities, the vector returned is of size 8*8*73 + 1 (for pass)
action_size = 8*8*73 + 1
'''
0 to 6: from h to a
7 to 13: from 1 to 8
14 to 20: from a to h
21 to 27: from 8 to 1
28 to 34: from h1 to a8
35 to 41: from a1 to h8
42 to 48: from a8 to h1
49 to 55: from h8 to a1
56 to 63: knight moves from 2left1down to 2down1left
64 to 66: pawn promotion to rook
67 to 69: pawn promotion to bishop
70 to 72: pawn promotion to knight
'''
