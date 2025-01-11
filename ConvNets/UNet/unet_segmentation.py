import os
import torch
import yaml
import numpy
# from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torchvision import models, datasets
from torchvision.transforms import ToTensor
from torch.nn.functional import relu
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.io import read_image

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.channels = [64, 128, 256, 512]
        self.encoder = Encoder(3, self.channels[:-1])
        self.conv1 = nn.Conv2d(self.channels[-1-1], self.channels[-1], kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(self.channels[-1], self.channels[-1], kernel_size=3, padding=1) 
        self.decoder = Decoder(self.channels[::-1])
        self.conv_out = nn.Conv2d(self.channels[0], n_classes, kernel_size=1)

    def forward(self, x):
        v, skips = self.encoder(x)
        v = self.conv1(v)
        v = self.conv2(v)
        v = self.decoder(v, skips)
        out = self.conv_out(v)
        return out


class Encoder(nn.Module):
    def __init__(self, n_feat, channels):
        super().__init__()
        self.channels = channels
        self.conv_blocks = nn.ModuleList()
        self.c_in = ConvBlock(n_feat, self.channels[0])
        for i in range(1, len(channels)):
            self.conv_blocks.append(ConvBlock(channels[i-1], channels[i]))
    
    def forward(self, x):
        skips = []
        v, skip = self.c_in(x)
        skips.append(skip)
        for conv in self.conv_blocks:
            v, skip = conv(v)
            skips.append(skip)

        skips = skips[::-1]
        return v, skips


class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.up_convs = nn.ModuleList()
        for i in range(1, len(self.channels)):
            self.up_convs.append(UpConvBlock(self.channels[i-1], self.channels[i]))

    def forward(self, x, skips):
        for i in range(len(self.channels)-1):
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
        # print(val.shape)
        return val, skip 
    
class UpConvBlock(nn.Module):
    def __init__(self, feat_in, feat_out):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(feat_in, feat_out, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(feat_in, feat_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feat_out, feat_out, kernel_size=3, padding=1)
    
    def forward(self, x, skip):
        upconv = self.conv_t(x)
        # print(f"output: {upconv.shape}")
        # print(f"skip: {skip.shape}")

        if upconv.shape[2:] != skip.shape[2:]:
            # print("resizing")
            upconv = F.interpolate(upconv, size=skip.shape[2:], mode="bilinear", align_corners=False)
        # if upconv.shape[2] != skip.shape[2] or upconv.shape[3] != skip.shape[3]:
        #     #center crop if the skip and output don't match before concatinating
        #     print("cropping...")
        #     _, _, h, w = skip.shape
        #     delta_h = (h - upconv.shape[2]) // 2
        #     delta_w = (w - upconv.shape[3]) // 2
        #     skip = skip[:, :, delta_h:delta_h + upconv.shape[2], delta_w:delta_w + upconv.shape[3]]
        val = torch.cat([upconv, skip], dim=1)
        val = relu(self.conv1(val))
        val = relu(self.conv2(val))
        return val

class CatDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None):
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.fnames = os.listdir(self.mask_dir)[:1000]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.fnames[idx])
        image = read_image(img_path)
        mask_path = os.path.join(self.mask_dir, self.fnames[idx])
        mask = read_image(mask_path)
        mask[mask > 0] = 1
        # print(f"image shape: {image.shape}")
        # print(f"mask shape: {mask.shape}")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image.float(), mask.float()

def get_data():
    with open("info.yaml", "r") as file:
        data = yaml.safe_load(file)
    img_p = data["img_path"]
    mask_p = data["mask_path"]
    dataset = CatDataset(img_p, mask_p)
    return dataset

if __name__ == "__main__":
    data = get_data()
    img, mask = data[0]
    
    train_count = int(0.8 * len(data))
    valid_count = len(data) - train_count
    # Randomly split the dataset
    train_dataset, val_dataset = random_split(data, [train_count, valid_count])

    # print(len(train_dataset))
    # print(len(val_dataset))
    # (Optional) Create DataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    # img, mask = data[0]
    
    # show mask
    # mask = img.numpy()
    # mask = mask.transpose(1,2,0)
    # print(mask.shape)
    # plt.imshow(mask)
    # plt.show()

    model = UNet(1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    num_epochs = 10
    model_path = "my_model.chk"

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data,target) in enumerate(train_loader):
            # print(data.shape)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print(f"{(batch_idx * len(data)):5}/{len(train_dataset):5}: {(((batch_idx*len(data))/len(train_dataset))*100):4}", end="\r")
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataset)}]\tLoss: {loss.item():.6f}')

        torch.save(model.state_dict(), model_path)
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
