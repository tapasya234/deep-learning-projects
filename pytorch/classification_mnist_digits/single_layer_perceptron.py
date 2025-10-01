from torch import nn


class SingleLayerPerceptron(nn.Module):
    def __init__(self, numClasses):
        super().__init__()

        self.numClasses = numClasses
        self.linear = nn.Linear(in_features=28 * 28, out_features=numClasses)

    def forward(self, x):
        # Flatten the (B, 1, 28, 28) tensor to (B, 784) features for the linear layer.
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
