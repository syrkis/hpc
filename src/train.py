# train.py
#   train vae model
# by: Noah Syrkis

# imports
import torch
import torch.nn.functional as F

from tqdm import tqdm


# train
def train(model, loader, optimizer, epochs, device):
    for epoch in range(epochs):
        pbar = tqdm(loader)
        losses = []
        for x, _ in pbar:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(x)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = F.binary_cross_entropy(x_hat, x, reduction='sum') + kld
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_description(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.2f}')
    return model
