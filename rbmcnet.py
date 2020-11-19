#!/usr/bin/env python

import torch
import torchvision 
import torch.nn as nn
import torch.nn.functional as F
import hyperparams
import time
import numpy as np
import os
from tqdm import tqdm
from fen_string_convert import convert_fen_string
import math

class ConvBlock(nn.Module):
    # Channels
    # P1 piece     6
    # P2 piece     6
    # Whose turn   1
    # P1 castling  2
    # P2 castling  2 
    #---------------
    #             17

    def __init__(self, in_dims):
        super(ConvBlock, self).__init__()
        self.channels, self.board_x, self.board_y = in_dims

        self.conv1 = nn.Conv2d(self.channels, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        s = s.view(-1, self.channels, self.board_x, self.board_y)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s



class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class OutBlock(nn.Module):
   # Action output space
   #
   # Location of piece piece to move = 8x8
   # 
   # Movement
   # 56 for "queen-like" moves
   #  8 for knight moves
   #  9 for underpromotions of pawns
   #---------
   # 73
   #
   # |Location| * |action| = 8*8*73 = 4672

    def __init__(self, input_dims, action_size):
        super(OutBlock, self).__init__()
        self.actions = action_size
        self.channels, self.board_x, self.board_y = input_dims

        self.conv = nn.Conv2d(128, 3, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3*8*8, 32)
        self.fc2 = nn.Linear(32, 1)
        
        self.conv1 = nn.Conv2d(128, 32, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(8*8*32, self.actions)
    
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 3*8*8)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))
        
        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 8*8*32)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v


class RbmcNet(nn.Module):

    # Input to the Rbmc ResNet is (17x8x8)
    # That is a 8x8 chess board with 17 planes.
    # Of these 17 planes:
    # - 6 are for the different P1 pieces
    # - 6 are for the difrerent P2 pieces
    # - 1 is to annotate whose turn it is
    # - 2 for P1 castling
    # - 2 for P2 castling
    # 
    # Output of this network has 2 heads.
    # A value head which is a real value between [-1, 1] which is our expected
    # result based on neural network evaluation of the current board state.
    # values close to -1 means prediction is we lose, 0 for tie, and 1 predicting we will win.
    # A policy head, which is a vector of 8*8*73 = 4672 probabilities indicating where the piece
    # we are to move is located and what action to take.

    def __init__(self, input_dims, action_size):
        super(RbmcNet, self).__init__()
        self.input_dims = input_dims
        self.conv = ConvBlock(input_dims)

        for block in range(19):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock(input_dims, action_size)
    
    def forward(self,s):
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s


# Neural network wrapper utilities from
# https://github.com/suragnair/alpha-zero-general

class NNetWrapper():
    def __init__(self):
        self.cuda = hyperparams.use_gpu and torch.cuda.is_available()
        self.batch_size = hyperparams.batch_size
        self.epochs = hyperparams.epochs
        self.channels, self.board_x, self.board_y = hyperparams.input_dims
        self.action_size = hyperparams.action_size
        self.nnet = RbmcNet(hyperparams.input_dims, hyperparams.action_size)
        self.device = "cuda:0"

        if self.cuda:
            self.nnet.to(self.device)

    def train(self, training_examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = torch.optim.Adam(self.nnet.parameters())

        examples = [(convert_fen_string(fen), pi, z) for (fen, pi, z) in training_examples]


        for epoch in range(self.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if self.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Save model
                filename = "epoch_" + str(epoch) + ".pth.tar"
                self.save_checkpoint(filename=filename)

        return total_loss

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if self.cuda: board = board.contiguous().to(self.device)

        # whom: what is this line for?  
        #board = board.view(1, self.board_x, self.board_y)

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * torch.log(outputs + 1e-10)) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
  
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if self.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])



class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return str(self.avg)
        #return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
