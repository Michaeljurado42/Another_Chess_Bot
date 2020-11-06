#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class IdBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, filters):
        super(IdBlock, self).__init__()
        self.in_channels = in_channels
        self.k = kernel_size
        self.f1, self.f2, self.f3 = filters
        
        self.conv1 = nn.Conv2d(self.in_channels, self.f1, kernel_size=(1,1), stride=(1,1))
        self.bn1 = nn.BatchNorm2d(self.f1)

        self.conv2 = nn.Conv2d(self.f1, self.f2, kernel_size=(kernel_size,kernel_size), stride=(1,1), padding=self.k//2)
        self.bn2 = nn.BatchNorm2d(self.f2)

        self.conv3 = nn.Conv2d(self.f2, self.f3, kernel_size=(1,1), stride=(1,1))
        self.bn3 = nn.BatchNorm2d(self.f3)

    def forward(self, x):
        shortcut = x

        # first component of main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # second component of main path
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        # third component of main path
        out = self.conv3(out)
        out = self.bn3(out)

        out = out + shortcut
        out = F.relu(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, filters, stride=2):
        super(ConvBlock, self).__init__()
    
        self.in_channels = in_channels
        self.k = kernel_size
        self.f1, self.f2, self.f3 = filters
        self.s = stride

        self.conv1 = nn.Conv2d(self.in_channels, self.f1, kernel_size=(1,1), stride=(stride, stride))
        self.bn1 = nn.BatchNorm2d(self.f1)

        self.conv2 = nn.Conv2d(self.f1, self.f2, kernel_size=(kernel_size, kernel_size), stride=(1,1), padding = kernel_size//2)
        self.bn2 = nn.BatchNorm2d(self.f2)

        self.conv3 = nn.Conv2d(self.f2, self.f3, kernel_size=(1,1), stride=(1,1))
        self.bn3 = nn.BatchNorm2d(self.f3)

        self.convx = nn.Conv2d(self.in_channels, self.f3, kernel_size=(1,1), stride=(stride, stride))
        self.bnx = nn.BatchNorm2d(self.f3)

    def forward(self, x):
        shortcut = x

        # main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # second component of main path
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        # third component of main path
        out = self.conv3(out)
        out = self.bn3(out)

        # shortcut path
        shortcut = self.convx(shortcut)
        shortcut = self.bnx(shortcut)

        out = out + shortcut
        out = F.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, input_shape):
        super(ResNet50, self).__init__()
        self.classes = 13

        self.conv1 = nn.Conv2d(input_shape[0], 64, 7, stride=(2,2), padding=3)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2_a = ConvBlock(64, kernel_size=3, filters=(64,64,256), stride = 1)
        self.conv2_b = IdBlock(256, kernel_size=3, filters=(64,64,256))
        self.conv2_c = IdBlock(256, kernel_size=3, filters=(64,64,256))

        self.conv3_a = ConvBlock(256, kernel_size=3, filters=(128,128,512), stride=2)
        self.conv3_b = IdBlock(512, kernel_size=3, filters=(128,128,512))
        self.conv3_c = IdBlock(512, kernel_size=3, filters=(128,128,512))
        self.conv3_d = IdBlock(512, kernel_size=3, filters=(128,128,512))

        self.conv4_a = ConvBlock(512, kernel_size=3, filters=(256,256,1024), stride=2)
        self.conv4_b = IdBlock(1024, kernel_size=3, filters=(256,256,1024))
        self.conv4_c = IdBlock(1024, kernel_size=3, filters=(256,256,1024))
        self.conv4_d = IdBlock(1024, kernel_size=3, filters=(256,256,1024))
        self.conv4_e = IdBlock(1024, kernel_size=3, filters=(256,256,1024))
        self.conv4_f = IdBlock(1024, kernel_size=3, filters=(256,256,1024))

        self.conv5_a = ConvBlock(1024, kernel_size=3, filters=(512,512,2048), stride=2)
        self.conv5_b = IdBlock(2048, kernel_size=3, filters=(512,512,2048))
        self.conv5_c = IdBlock(2048, kernel_size=3, filters=(512,512,2048))

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(2048, self.classes)


    def forward(self, x):
        # Stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        # Stage 2
        x = self.conv2_a(x)
        x = self.conv2_b(x)
        x = self.conv2_c(x)

        # Stage 3
        x = self.conv3_a(x)
        x = self.conv3_b(x)
        x = self.conv3_c(x)
        x = self.conv3_d(x)

        # Stage 4
        x = self.conv4_a(x)
        x = self.conv4_b(x)
        x = self.conv4_c(x)
        x = self.conv4_d(x)
        x = self.conv4_e(x)
        x = self.conv4_f(x)

        # Stage 5
        x = self.conv5_a(x)
        x = self.conv5_b(x)
        x = self.conv5_c(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x




