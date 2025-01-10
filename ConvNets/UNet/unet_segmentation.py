import torch
import numpy
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torchvision import models, datasets
from torchvision.transforms import ToTensor
from torch.nn.functional import relu
from torch.utils.data import DataLoader

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = Encoder(1)
        self.conv1 = nn.Conv2d(112, 224, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(224, 224, kernel_size=3, padding=1) 
        self.decoder = Decoder()
        self.conv_out = nn.Conv2d(56, n_classes, kernel_size=1)

    def forward(self, x):
        v, skips = self.encoder(x)
        v = self.conv1(v)
        v = self.conv2(v)
        v = self.decoder(v, skips)
        out = self.conv_out(v)
        return out


class Encoder(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.c1 = ConvBlock(n_feat, 56) # 64
        self.c2 = ConvBlock(56, 112) #128
        # self.c3 = ConvBlock(128, 256)
        # self.c4 = ConvBlock(256, 512)
    
    def forward(self, x):
        skips = []
        v, out1 = self.c1(x)
        v, out2 = self.c2(v)
        # v, out3 = self.c3(v)
        # v, out4 = self.c4(v)
        
        # skips.append(out4)
        # skips.append(out3)
        skips.append(out2)
        skips.append(out1)

        return v, skips


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        features = 224
        self.up_convs = nn.ModuleList()
        for i in range(2):
            out = int(features/2)
            self.up_convs.append(UpConvBlock(features, out))
            features = out

    def forward(self, x, skips):
        for i in range(2):
            x = self.up_convs[i](x, skips[i])
        return x

class ConvBlock(nn.Module):
    def __init__(self, feat_in, feat_out):
        super().__init__()
        self.conv1 = nn.Conv2d(feat_in, feat_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feat_out, feat_out, kernel_size=3, padding=1)
        self.pool= nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = relu(self.conv1(x))
        skip = relu(self.conv2(out))
        val = self.pool(skip)
        print(val.shape)
        return val, skip 
    
class UpConvBlock(nn.Module):
    def __init__(self, feat_in, feat_out):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(feat_in, feat_out, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(feat_in, feat_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feat_out, feat_out, kernel_size=3, padding=1)
    
    def forward(self, x, skip):
        upconv = self.conv_t(x)
        print(f"output: {upconv.shape}")
        print(f"skip: {skip.shape}")
        val = torch.cat([upconv, skip], dim=1)
        val = relu(self.conv1(val))
        val = relu(self.conv2(val))
        return val

if __name__ == "__main__":
    # Load the MNIST dataset
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    val_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    model = UNet(1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data,target) in enumerate(train_loader):
            print(data.shape)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_data.dataset)}]\tLoss: {loss.item():.6f}')

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_data:
                output = model(data)
                val_loss += criterion(output, target).item()  # Sum up batch loss

        val_loss /= len(val_data)
        print(f'Validation Epoch: {epoch}\tLoss: {val_loss:.6f}')








    # labels_map = {
    #     1: "1",
    #     2: "2",
    #     3: "3",
    #     4: "4",
    #     5: "5",
    #     6: "6",
    #     7: "7",
    #     8: "8",
    #     9: "9",
    #     0: "0",
    # }

    # figure = plt.figure(figsize=(28, 28))
    # cols, rows = 3, 3
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(train_data), size=(1,)).item()
    #     img, label = train_data[sample_idx]
    #     figure.add_subplot(rows, cols, i)
    #     plt.title(labels_map[label])
    #     plt.axis("off")
    #     plt.imshow(img.squeeze(), cmap="gray")
    # plt.show()
