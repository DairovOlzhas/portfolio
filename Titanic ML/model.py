import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(7, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, input):

        x = self.fc1(input)
        x = torch.sigmoid(x)

        x = self.fc2(x)
        x = torch.sigmoid(x)

        out = self.fc3(x)
        out = torch.sigmoid(out)

        return out
