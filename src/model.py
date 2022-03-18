# model.py
#   pytorch model
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn


# model class definition
class Model(nn.Module):

    def __init__(self): # depending on model give vocab size or sample length as init input
        self.fc1 = nn.Linear(50, 10)
        self.fc2 = nn.Linear(10, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        x = self.forward(x) 
        x = torch.arg(x, dim=1)   
        return x


# call stack
def main():
    model = Model()

if __name__ == "__main__":
    main()
