#!/usr/bin/env python
#%%
import torch
import torchvision
from resnet50 import *
from rbmcnet import *

#%%

device = torch.device("cuda:0")

#model = ResNet50((119,224,224), classes=40)
#model = ResNet50((13,8,8), classes=40)

model = RbmcNet()

model.to(device)


# %%
