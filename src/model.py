# model.py
#   vae model for mnist   
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn


# VAE model
class Model(nn.Module):
    def __init__(self, latent_dim):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc21 = nn.Linear(128, latent_dim)
        self.fc22 = nn.Linear(128, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, 9216)
        self.deconv1 = nn.ConvTranspose2d(16, 8, 3, 1)
        self.deconv2 = nn.ConvTranspose2d(8, 1, 3, 1)

    def encode(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        mu = self.fc21(x)
        sigma = self.fc22(x)
        return mu, sigma

    def decode(self, x):
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = x.view(x.shape[0], 16, 24, 24)
        x = self.deconv1(x)
        x = torch.relu(x)
        x = self.deconv2(x)
        x = torch.sigmoid(x)
        return x

    def reparameaterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) * .5
        return std * eps + mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameaterize(mu, logvar)
        return self.decode(z), mu, logvar
    
