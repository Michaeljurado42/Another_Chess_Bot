#!/usr/bin/env python

import torch
import torchvision 
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    # Channels
    # P1 piece     6
    # P2 piece     6
    # Whose turn   1
    # P1 castling  2
    # P2 castling  2 
    #---------------
    #             17

    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(17, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        s = s.view(-1, 17, 8, 8)  # batch_size x channels x board_x x board_y
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

    def __init__(self):
        super(OutBlock, self).__init__()
        self.actions = 4672

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

    def __init__(self):
        super(RbmcNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(19):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()
    
    def forward(self,s):
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s


