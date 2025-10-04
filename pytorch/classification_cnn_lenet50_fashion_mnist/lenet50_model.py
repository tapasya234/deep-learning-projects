from turtle import forward
from torch import nn

# LeNet5 network has two convolutional layers and three fully connected layers.
# The model architecture can be then divided into two parts:
# _body: This implements the Convolutional part of the network, which consists of
# the two convolutional layers. Each convolutional layer is followed by a pooling layer.
# In classical Machine Learning terms, it also works as a feature extractor.

# _head: This implements the fully-connected part of the network. It consists of
# three fully-connected layers, with the last layer having ten output classes.

# Note: There is no activation after the last Linear layer.
# So while during inference, if we want probability as the output, we need to
# pass the model output through softmax.
# Note: This implementation of LeNet5 Model is different from the original version
# which uses Average Pooling and Tanh Activation but this model uses
# ReLU Activation and Max Pooling.


class LeNet5(nn.Module):
    def __init__(self, numClasses):
        super().__init__()

        # Convolution Layers
        self._body = nn.Sequential(
            # First Convolution Layer.
            #  Input size - (32, 32). Output Size - (28, 28)
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            # ReLU Activation Layer.
            nn.ReLU(),
            # Max Pool 2-d.
            nn.MaxPool2d(kernel_size=2),
            # Second Convolution Layer.
            # Input size - (14,14) Output size - (10,10)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Output size - (5, 5)
        )

        # Fully connected layers.
        self._head = nn.Sequential(
            # First fully connected layer.
            # in_features = total number of weights in the previous convolution layer.
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            # ReLU Activation Layer.
            nn.ReLU(inplace=True),
            # Second fully connected layer.
            # in_features = output of the last linear year.
            nn.Linear(in_features=120, out_features=84),
            # ReLU Activation Layer.
            nn.ReLU(inplace=True),
            # Third fully connected layer.
            # in_features = Output of the last linear layer.
            # out_features = Number of classes in the dataset.
            nn.Linear(in_features=84, out_features=numClasses),
        )

    def forward(self, x):
        # Apply feature extractor.
        x = self._body(x)

        # Flatten the output of the convolution layer.
        # Dimension should be batchSize * num of layers in the last convolution layer.
        x = x.view(x.shape[0], -1)

        # Apply classification head.
        x = self._head(x)

        return x
