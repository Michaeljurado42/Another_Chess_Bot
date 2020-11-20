#!/usr/bin/env python
# coding: utf-8

# # Overview
# this is a training script for the RNN. To train against multiple types of agents, edit the run_game method.
# In order for this program to work record_data must be working

# In[1]:


import torch
import pickle
from uncertainty_rnn import BoardGuesserNet
import torch.optim as optim
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from collections import deque
import sys
from modified_play_game import play_game
from fen_string_convert import convert_truncated_to_truth

# # Board Guess

# In[2]:


# set up Board Guesser Net
guessNet_white = BoardGuesserNet()
guessNet_black = BoardGuesserNet()

# Constants
train_iterations = 100000
validation_count = 50  # after every 20 games check validation score
load_in_weights = True

# decayed_learning_rate = learning_rate *
#                        decay_rate ^ (global_step / decay_steps)
decay_steps = 100
decay_rate = .99
initial_lr = .00002
minimum_lr = .000005

# set up loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer1 = optim.Adam(guessNet_white.parameters(), lr=initial_lr)

optimizer2 = optim.Adam(guessNet_black.parameters(), lr=initial_lr)

# # Definer helpers

# In[3]:


def save_model(guessNet, white):
    # definer helpers
    try:
        os.mkdir("rnn_model")
    except Exception as e:
        print(e)
    if white:
        torch.save(guessNet.state_dict(), "white_rnn_model")
    else:
        torch.save(guessNet.state_dict(), "black_rnn_model")


def load_model(guessNet, white):
    if white:
        guessNet.load_state_dict(torch.load("white_rnn_model"))
    else:
        guessNet.load_state_dict(torch.load("black_rnn_model"))


def run_game():
    """
    Returns X_train_batch, y_train_batch, which are both numpy arrays
    
    NOTE: Addit command to change which agent plays. (Must save data in specified format though)
    """
    white_data, black_data = play_game("random_agent_save_obs", "random_agent_save_obs")

    white_sense_list, white_truth_board_list = white_data

    black_sense_list, black_truth_board_list = black_data

    X_train_batch_white = np.array(white_sense_list)
    y_train_batch_white = np.array(white_truth_board_list)

    X_train_batch_black = np.array(black_sense_list)
    y_train_batch_black = np.array(black_truth_board_list)

    return X_train_batch_white, y_train_batch_white, X_train_batch_black, y_train_batch_black


def create_loss_plot(train_loss_history, white):
    """

    :param train_loss_history: list of loss history
    :return:
    """
    plt.figure()

    x_axis = (np.arange(len(train_loss_history)))

    plt.plot(x_axis, train_loss_history, label="train_loss")

    plt.legend(loc="upper left")
    plt.ylabel("Mean CategoricalCrossEntropyLoss")
    plt.xlabel("epochs")

    if white:
        save_name = "white_rnn_loss.png"
    else:
        save_name = "black_rnn_loss.png"
    try:
        plt.savefig(save_name)
    except Exception as e:
        print("Permission denied saving loss")

    plt.cla()
    plt.clf()
    plt.close()

def train_step(X_train_batch, y_train_batch, guessNet, optimizer, train = True):
    """
    Returns loss
    :param X_train_batch:
    :param y_train_batch:
    :param guessNet:
    :return:
    """
    X_train_batch = torch.Tensor(X_train_batch)

    # training step
    pred_labels = guessNet(X_train_batch)
    y_train_batch = torch.Tensor(y_train_batch)

    stacked_pred = torch.cat([i for i in pred_labels], axis=0)
    stacked_truth = torch.cat([i for i in y_train_batch], axis=0).argmax(1)
    loss = criterion(stacked_pred, stacked_truth)

    if train:
        loss.backward()
        optimizer.step()
    # magic
    return loss

def check_first_square(X_train_batch, guessNet):
    """
     prints out the predicted board at timestep 1. Makes sure the neural net actually learns something

    :param X_train_batch:
    :param y_train_batch:
    :param guessNet:
    :return:
    """
    X_train_batch = torch.Tensor(X_train_batch)
    pred_labels = guessNet(X_train_batch)
    first_pred_label = pred_labels[0].detach().cpu().numpy()

    # take an argmax to get the most probable board
    max_pred = np.zeros(first_pred_label.shape)
    max_pred[np.arange(first_pred_label.shape[0]), np.argmax(first_pred_label, axis=1)] = 1

    # convert it into standard truth board format
    pred_board = convert_truncated_to_truth(max_pred)

    print("pred board for first time step")
    index = 0
    for channel in [np.argwhere(i) for i in pred_board]:
        print("active arguments in channel", index)
        print(channel)
        index += 1



# In[9]:

def decay_learning_rate(optimizer, epoch):
    """

    :param optimizer:
    :param epoch:
    :return:
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(initial_lr * decay_rate ** (epoch / decay_steps), minimum_lr)

    print("decaying learning rate to", param_group["lr"])


if load_in_weights:

    load_model(guessNet_white, white=True)
    load_model(guessNet_black, white=False)


# contains train and test lost history
train_loss_history_white = []
train_loss_history_black = []
test_loss_history = []
plt.figure()

train_loss_queue_white = deque(maxlen=100)  # holds a temporary queue of train lost
train_loss_queue_black = deque(maxlen=100)
smallestLoss_white = sys.maxsize
smallestLoss_black = sys.maxsize

lastImprovement = 0
for epoch in range(train_iterations):
    X_train_batch_white, y_train_batch_white, X_train_batch_black, y_train_batch_black = run_game()  # plays a game random versus random

    # Run training steps for white then black
    loss_white = train_step(X_train_batch_white, y_train_batch_white,guessNet_white, optimizer1)
    loss_black = train_step(X_train_batch_black, y_train_batch_black, guessNet_black, optimizer2)

    train_loss_queue_white.append(loss_white.detach().cpu().numpy())
    train_loss_queue_black.append(loss_black.detach().cpu().numpy())

    # take running average to make learning curve smoother
    loss = np.mean(train_loss_queue_white)
    train_loss_history_white.append(np.mean(train_loss_queue_white))

    loss = np.mean(train_loss_queue_black)
    train_loss_history_black.append(np.mean(train_loss_queue_black))

    if (epoch + 1) % 5 == 0:  # save loss plot every 5 steps
        create_loss_plot(train_loss_history_white, white=True)
        create_loss_plot(train_loss_history_black, white=False)

    # run a validation every 100 steps. If loss is better save this model
    if (epoch + 1) % 100 == 0:
        print("validation time")
        total_val_loss_white = 0
        total_val_loss_black = 0
        for i in range(validation_count):
            X_train_batch_white, y_train_batch_white, X_train_batch_black, y_train_batch_black = run_game()  # plays a game random versus random

            # Run training steps
            loss_white = train_step(X_train_batch_white, y_train_batch_white, guessNet_white, optimizer1, train=False)
            loss_black = train_step(X_train_batch_black, y_train_batch_black, guessNet_black, optimizer2, train=False)

            total_val_loss_white += loss_white
            total_val_loss_black += loss_black
        val_loss_white = total_val_loss_white / validation_count
        val_loss_black = total_val_loss_black / validation_count

        if val_loss_white < smallestLoss_white:
            print("better model found with loss", val_loss_white, "for white")
            save_model(guessNet_white, white=True)
            smallestLoss_white = val_loss_white

        if val_loss_black < smallestLoss_black:
            print("better model found with loss", val_loss_black, "for black")
            save_model(guessNet_black, white=False)
            smallestLoss_black = val_loss_black

    # # decay learning rate
    if epoch > 200:
        decay_learning_rate(optimizer1, epoch)
        decay_learning_rate(optimizer2, epoch)

    # verify that neural network is at least predicting initial board correctly
    if (epoch + 1) % 100 == 0:

        X_train_batch_white, y_train_batch_white, X_train_batch_black, y_train_batch_black = run_game()
        print("printing board for white")
        check_first_square(X_train_batch_white,guessNet_white)

