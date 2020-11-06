"""Store models here"""
import torch
from torch import nn


class SmallValueNet(nn.Module):

    def __init__(self):
        super(SmallValueNet, self).__init__()

        """This could be a small backbone"""
        self.conv1 = torch.nn.Conv2d(12, 32, 3)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2)

        self.conv2 = torch.nn.Conv2d(32, 64, 2)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2)

        self.flatten = torch.nn.Flatten()

        """maybe this could be where the network splits"""
        self.dense1 = torch.nn.Linear(64, 128)
        self.relu3 = torch.nn.ReLU()

        self.value = torch.nn.Linear(128, 1)

    def forward(self, state):
        conv_1_out = self.conv1(state)
        print("conv 1 out shape", conv_1_out.shape)

        relu_1 = self.relu1(conv_1_out)
        pool_1_out = self.pool1(relu_1)
        print("pool_1_out shape", pool_1_out.shape)

        conv_2_out = self.conv2(pool_1_out)
        print("conv 2 out shape", conv_2_out.shape)

        relu_2 = self.relu2(conv_2_out)
        pool_2_out = self.pool2(relu_2)
        print("pool_2_out shape", pool_2_out.shape)

        flatten_out = self.flatten(pool_2_out)
        print("flatten shape", flatten_out.shape)

        dense1_out = self.dense1(flatten_out)
        relu_3 = self.relu3(dense1_out)

        return self.value(relu_3)

class LargeValueNet(nn.Module):
    """ Res Network. """
    pass


if __name__ == "__main__":
    """Sample test residual network"""
    value_net = SmallValueNet()
    sample_state = torch.ones((1, 12, 8, 8))
    probaility_of_winning = value_net(sample_state)
    print(probaility_of_winning)

