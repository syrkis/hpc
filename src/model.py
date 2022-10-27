# model.py
#   vae model for mnist   
# by: Noah Syrkis

# imports
import torch
from torch import nn


# VAE model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(18432, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 18432)
        self.deconv1 = nn.ConvTranspose2d(32, 16, 3, 1)
        self.deconv2 = nn.ConvTranspose2d(16, 1, 3, 1)

    def encode(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

    def decode(self, x):
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = x.view(x.shape[0], 32, 24, 24)
        x = self.deconv1(x)
        x = torch.relu(x)
        x = self.deconv2(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
