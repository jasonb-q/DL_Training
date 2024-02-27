import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import flatten
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchio import ScalarImage


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
        self.sm = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.pool(x)

        x = flatten(x, 1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sm(x)
        return x

class MGMTDataset(Dataset):
    """
    The PyTorch dataset for MGMT classification.
    """
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = ScalarImage(self.df.iloc[idx]["T1"])
        label = self.df.iloc[idx]["MGMT"]
        return image, label
    

def read_data(data_path):
    """
    Load the data from a csv file.
    """
    data = pd.read_csv(data_path)
    msk = np.random.rand(len(data)) < 0.8
    train = data[msk]
    test = data[~msk]
    return train, test

def custom_mse_loss(y_true, y_pred):
    return torch.mean(torch.square(y_true - y_pred))

if __name__ == "__main__":
    """
    Basic model for potential use in MGMT classification. This is for learning purposes.
    """
    # Image size is 240x240
    # Load the data
    data_path = "data/brats_2021.csv"
    train, test = read_data(data_path)
    train_dataset = MGMTDataset(train)
    test_dataset = MGMTDataset(test)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)

    # Create the model
    net = MGMTModel()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    for epoch in range(2):  # loop over the dataset multiple times
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            # loss = criterion(outputs, labels)
            loss = custom_mse_loss(outputs, labels)
            loss.backward()
            optimizer.step()