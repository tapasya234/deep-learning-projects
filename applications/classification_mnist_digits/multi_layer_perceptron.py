from torch import nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, numClasses):
        super().__init__()

        self.numClasses = numClasses

        self.classifier = nn.Sequential(
            # Input Layer
            nn.Linear(in_features=28 * 28, out_features=512),
            # ReLU activation
            nn.ReLU(),
            # Hidden Layer
            nn.Linear(in_features=512, out_features=512),
            # ReLU activation
            nn.ReLU(),
            # Output Layer
            nn.Linear(in_features=512, out_features=self.numClasses),
        )

    def forward(self, x):
        # Flatten (B,1,28,28) to (B,784) features for the Linear layer.
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
