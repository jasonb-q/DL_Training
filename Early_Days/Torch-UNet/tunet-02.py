import torch
from torch import nn, optim
from torch import flatten
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchio import ScalarImage
from typing import Tuple, List, NoReturn
from torchvision.transforms import CenterCrop
from utils import load_settings
import nibabel as nib
from torch.utils.data import DataLoader, Dataset
import sys

settings = load_settings(sys.argv[1])
print(settings["dimension"])
device = ('cuda' if torch.cuda.is_available() else 'cpu')

class Block(nn.Module):
    """
    The PyTorch model for JUNet classification.
    """
    def __init__(self, input_channels: int, output_channels: int):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)

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
    def __init__(self, channels: tuple):
        super(Encoder, self).__init__()

        # convolutions
        self.enc_blocks = nn.ModuleList(
            [Block(channels[i], channels[i+1]) for i in range(len(channels) - 1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        block_outputs = []
        # print(x.shape)
        block_outputs.append(x)
        x = self.pool(x)
        for block in self.enc_blocks:
            x = block(x) # perform the convolutions and ReLU
            block_outputs.append(x)
            x = self.pool(x)

        for block in block_outputs:
            print(block.shape)
        return x, block_outputs
    
    
class Decoder(nn.Module):
    """
    The PyTorch decoder for JUNet classification.
    """
    def __init__(self, channels: Tuple[int]):
        super(Decoder, self).__init__()
        self.channels = channels

        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels) - 1)] # kernel 2 and stride 2
        )

    def forward(self, x, blocks):
        print(x.shape)
        for block in blocks:
            print(block.shape)
        print("Upconvolution")
        for i in range(len(self.channels)-1):
            print(f"pre-upcon: {x.shape}")
            x = self.upconvs[i](x)
            # print(x.shape)
            print(f"concat dim {x.shape} and {blocks[i].shape}")
            x = torch.cat([x, blocks[i]], dim=1)
            print(f"After Concat dim {x.shape}")
            
        return x
    

class UNet(nn.Module):
    """
    The PyTorch UNet model for JUNet classification.
    """
    def __init__(self):
        super(UNet, self).__init__()
        self.enc_channels: Tuple[int] = tuple(settings["channels"]) # 16, 32, 64, 128, 256 
        self.dec_channels: Tuple[int] = self.enc_channels[::-1] # 256, 128, 64, 32, 16 
        print(self.enc_channels)
        print(self.dec_channels)
        self.num_classes: int = settings["num_classes"] # The number of classes to classify
        self.out_size: Tuple[int] = tuple(settings["dimension"])  # if retain_dim is True, then the output size will be 128x128 using interpolation

        self.initial_conv = Block(1, self.enc_channels[0])
        # The encoder and decoder for the UNet model. 
        self.encoder = Encoder(self.enc_channels)
        self.decoder = Decoder(self.dec_channels)
        
        self.final_conv = nn.Conv2d(self.dec_channels[-1], self.num_classes, 1)
        self.final_activation = nn.Softmax()


    def forward(self, x):
        print("Ecoder")
        print(x.shape)
        x = self.initial_conv(x)
        # encode
        x, blocks = self.encoder(x)

        # decode
        print("Decoder")
        decode_features = self.decoder(blocks[::-1][0], blocks[::-1][1:]) 

        map = self.final_conv(decode_features)
        map = self.final_activation(map)
        return map
    
#-----Define train function-----#
def train(model, train_generator, val_generator, loss_fxn, optimizer, n_epochs):
    # model.train()
    for epoch in range(n_epochs):
        epoch_loss = []
        for batch_idx, (data, target) in enumerate(train_generator):
            data, target = data.to(device), target.to(device)  # Move to device
            optimizer.zero_grad()  # Clear existing gradients
            output = model(data)  # Forward pass
            loss = loss_fxn(output, target)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters
            epoch_loss.append(loss.item())
            
        print(f'Epoch: {epoch}, Loss: {np.mean(np.array(epoch_loss))}')


#-----Create data generators for train and validation data-----#
def data_generator(train, val, params):
    training_generator = torch.utils.data.DataLoader(train, **params)
    validation_generator = torch.utils.data.DataLoader(val, **params)
    return training_generator, validation_generator


#-----Define DSC loss function as a neural net module-----#
class DSCLoss(nn.Module):
    def __init__(self):
        super(DSCLoss, self).__init__()
        self.flatten = nn.Flatten()

    def forward(self, inputs, target):
        smooth = 1e-8
        logits = self.flatten(inputs)
        target = self.flatten(target)

        intersection = (logits*target).sum()
        dice = (2. * intersection + smooth) / (target.sum() + logits.sum() + smooth)
        return 1 - dice


# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

#-----Initialize device-----#
    
if __name__ == "__main__":
    """UNet model using torch for classifying brain images in some way."""
    model = UNet().to(device)
    # Creating dummy data as per specifications
    X_data = np.zeros((1000, 1, 192, 192), dtype=np.float32)
    Y_data = np.ones((1000, 2, 192, 192), dtype=np.float32)

    # Initialize Dataset
    dataset = CustomDataset(X_data, Y_data)

    # DataLoader
    train_generator = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True)
    val_generator = train_generator

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    dsc_loss = DSCLoss()
    train(model, train_generator, val_generator, dsc_loss, optimizer, n_epochs=settings["epochs"])
