import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import flatten
import numpy as np
import pandas as pd


class MGMTModel(nn.Module):
    """
    The PyTorch model for MGMT classification.
    """
    def __init__(self, input_size, output_size):
        super(MGMTModel, self).__init__()
        # 240x240 image
        self.conv1 = nn.Conv2d(1, 8, 5, padding='same')
        # 240x240 mp = 120x120
        self.pool = nn.MaxPool2d(2, 2)
        # 120x120 image
        self.conv2 = nn.Conv2d(8, 16, 5, padding='same') 
        # 120X120 use 16 featues * 120x120 image
        self.fc1 = nn.Linear(16 * 120 * 120, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10) # output layer

    def forward(self, x):
        x = self.pool(F.softmax(self.conv1(x)))
        x = self.pool(F.softmax(self.conv2(x)))
        x = flatten(x, 1) # flatten all dimensions except batch
        x = F.softmax(self.fc1(x))
        x = F.softmax(self.fc2(x))
        x = self.fc3(x)
        return x



if __name__ == "__main__":
    """
    Basic model for potential use in MGMT classification. This is for learning purposes.
    """
    # Image size is 240x240
    # Load the data
    data = pd.read_csv("data/brats_2021.csv")
    msk = np.random.rand(len(data)) < 0.8
    train = data[msk]
    test = data[~msk]
    print(data.head())
    print(train.head())
    print(test.head())
    print(len(train), len(test))
