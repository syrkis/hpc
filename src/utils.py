# utils.py
#   useful functions for the project
# by: Noah Syrkis

# imports
from torchvision import datasets, transforms

import numpy as np

from matplotlib import pyplot as plt

from argparse import ArgumentParser



# functions
def load_data():
    data = datasets.MNIST('data', train=True, download=True, 
                          transform=transforms.ToTensor())
    return data

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--generate', action='store_true', default=False)
    args = parser.parse_args()
    return args

def plot(imgs, n):
    imgs = 1 - imgs
    dim = int(np.sqrt(n))
    fig, axes = plt.subplots(dim, dim, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i].reshape(28, 28), cmap='binary')
        ax.set(xticks=[], yticks=[])
        # remove ax edges
        for edge, spine in ax.spines.items():
            spine.set_visible(False)
    # black background
    fig.patch.set_facecolor('black')
    plt.show()












