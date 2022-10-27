# train.py
#   train vae model
# by: Noah Syrkis

# imports
from tqdm import tqdm


# train
def train(model, loader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        for x, _ in tqdm(loader):
            # x = x.view(x.shape[0], -1)
            optimizer.zero_grad()
            x_hat = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
    return model
