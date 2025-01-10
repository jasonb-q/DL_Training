import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import flatten
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchio import ScalarImage
from typing import Tuple, List, NoReturn
from torchvision.transforms import CenterCrop
from utils import load_settings
import sys

settings = load_settings(sys.argv[1])
print(settings["dimension"])

class Block(nn.Module):
    """
    The PyTorch model for JUNet classification.
    """
    def __init__(self, input_channels: int, output_channels: int):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    """
    The PyTorch encoder for JUNet classification.
    """
    def __init__(self, channels: tuplechannels: tuple):
        super(Encoder, self).__init__()

        # convolutions
        self.enc_blocks = nn.ModuleList(
            [Block(channels[i], channels[i+1]) for i in range(len(channels) - 1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        block_outputs = []
        for block in self.enc_blocks:
            x = block(x) # perform the convolutions and ReLU
            block_outputs.append(x)
            x = self.pool(x)
        return block_outputs
    
    
class Decoder(nn.Module):
    """
    The PyTorch decoder for JUNet classification.
    """
    def __init__(self, channels: Tuple[int]):
        super(Decoder, self).__init__()
        self.channels = channels

        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels) - 1)]
        )

        self.dec_blocks = nn.ModuleList(
            [Block(channels[i], channels[i+1]) for i in range(len(channels) - 1)]
        )

    def forward(self, x, enc_features):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            if x[:-1] != enc_features[:-1]:
                raise Exception("enc_features dimension != x dimension")
            x = torch.cat([x, enc_features[i]], dim=1)
            x = self.dec_blocks[i](x)
        return x
    

class UNet(nn.Module):
    """
    The PyTorch UNet model for JUNet classification.
    """
    def __init__(self):
        super(UNet, self).__init__()
        self.enc_channels: Tuple[int] = tuple(settings["channels"]) # 16, 32, 64, 128, 256 
        self.dec_channels: Tuple[int] = self.enc_channels[::-1] # 256, 128, 64, 32, 16 
        self.retain_dim: bool = settings["retain_dim"]
        self.num_classes: int = settings["num_classes"] # The number of classes to classify
        self.out_size: Tuple[int] = tuple(settings["dimension"])  # if retain_dim is True, then the output size will be 128x128 using interpolation

        # The encoder and decoder for the UNet model. 
        self.encoder = Encoder(self.enc_channels)
        self.decoder = Decoder(self.dec_channels)
        
        self.head = nn.Conv2d(self.dec_channels[-1], self.num_classes, 1)


    def forward(self, x):
        # encode
        enc_features = self.encoder(x)

        # decode
        decode_features = self.decoder(enc_features[::-1][0], enc_features[::-1][1:]) 

        map = self.head(decode_features)

        if self.retain_dim:
            map = F.interpolate(map, self.out_size)
        return map
    



if __name__ == "__main__":
    """UNet model using torch for classifying brain images in some way."""
    net = UNet()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    for epoch in range(2):  # loop over the dataset multiple times
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss.backward()
            optimizer.step()
