import torch
from torch import nn


class LogisticRegressionModel(nn.Module):
    def __init__(self, nFeatures):
        super().__init__()
        self.linear = nn.Linear(in_features=nFeatures, out_features=1, bias=True)

    def forward(self, X):
        X = self.linear(X)

        return torch.sigmoid(X)
