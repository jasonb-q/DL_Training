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
        # why not use ReLU here?
        # because we are going from output_channels to output_channels after using ReLU so there shouldn't be any negative values?
        return x


class Encoder(nn.Module):
    """
    The PyTorch encoder for JUNet classification.
    """
    def __init__(self, channels: Tuple[int] = (3, 16, 32, 64)):
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
            """Why do they append the output of the block to block_outputs before pooling? 
            The only reason I can see is that they want to keep the output of the block before pooling and the output of pooling
            is used somewhere else somehow.

            They are re-assigning x so I don't think x is being updated in the list(I could be wrong though).
            from my perspective the pooling part isn't being used.
            """
            block_outputs.append(x)
            x = self.pool(x)
        return block_outputs
    
    
class Decoder(nn.Module):
    """
    The PyTorch decoder for JUNet classification.
    """
    def __init__(self, channels: Tuple[int] = (64, 32, 16)):
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
            enc_feat = self.crop(enc_features[i], x)
            x = torch.cat([x, enc_feat], dim=1)
            x = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_feat, x):
        _, _, H, W = x.shape
        enc_feat = CenterCrop([H, W])(enc_feat)
        return enc_feat
    

class UNet(nn.Module):
    """
    The PyTorch UNet model for JUNet classification.
    """
    def __init__(self):
        super(UNet, self).__init__()
        """
        The encoding channels for the UNet model. 
        So the first conv layer will have 3 input channels and 16 output channels.
        The second conv layer will have 16 input channels and 32 output channels.
        The third conv layer will have 32 input channels and 64 output channels.
        """
        self.enc_channels = (3, 16, 32, 64)
        self.dec_channels = (64, 32, 16) # then upconv will go from 16 to 32 to 64
        self.retain_dim = True
        self.num_classes = 1 # The number of classes to classify
        self.out_size = (128, 128)  # if retain_dim is True, then the output size will be 128x128 using interpolation

        # The encoder and decoder for the UNet model. 
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        self.head = nn.Conv2d(self.dec_channels[-1], self.num_classes, 1)


    def forward(self, x):
        # encode
        enc_features = self.encoder(x)

        # decode
        decode_features = self.decoder(enc_features[::-1][0], enc_features[::-1][1:]) # confused here

        map = self.head(decode_features)

        if self.retain_dim:
            map = F.interpolate(map, self.out_size)
        return map
    
if __name__ == "__main__":
    """UNet model using torch for classifying brain images in some way."""
