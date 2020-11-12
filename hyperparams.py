#!/usr/bin/env python

# Run with GPU.  Set this to false if 
# getting runtime errors on version python 3.5
# Seems to run fine for python 3.7.9
use_gpu = True

# Variables for training neural network
batch_size = 128
epochs = 20

# Input dimensions of canonical board that is fed into neural network
input_dims = (20, 8, 8)

# For the action probabilities, the vector returned is of size 8*8*73 + 1 (for pass)
action_size = 8*8*73 + 1

