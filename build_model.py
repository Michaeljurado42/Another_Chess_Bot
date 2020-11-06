#!/usr/bin/env python
#%%
import torch
import torchvision
from resnet50 import *
from rbmcnet import *

#%%

device = torch.device("cpu")

#model = ResNet50((119,224,224), classes=40)
#model = ResNet50((13,8,8), classes=40)

model = RbmcNet()

model.to(device)


# [batch_size, channels, height, width]
x = torch.empty(1, 17,8, 8)  
x = x.to(device)

model

# %%
x = x.to(device)
x

# %%
model.forward(x)

# %%
