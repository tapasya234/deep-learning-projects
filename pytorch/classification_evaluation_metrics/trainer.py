import torch
from torch import nn
from torch import optim


# Since PyTorch does not provide unified methods for training,
# create a simple Trainer class to fit the model and make predictions.
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimiser: optim.Optimizer,
        numEpochs: int,
    ):
        """
        Docstring for __init__

        :param self: Description
        :param model: The `model` that needs to be trained.
        :type model: nn.Module
        :param criterion: The loss function used to train the model.
        :type criterion: nn.Module
        :param optimiser: The optimiser
        :type optimiser: optim.Optimizer
        :param numEpochs: The number of epochs used for training.
        :type numEpochs: int
        """
        self.model = model
        self.criterion = criterion
        self.optimiser = optimiser
        self.numEpochs = numEpochs

    def fit(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        fit updates model trainable parameters in a loop for given number of epochs.

        :param self: Description
        :param inputs: The inputs used to train the model.
        :type inputs: torch.Tensor
        :param targets: The actual classification of the inputs.
        :type targets: torch.Tensor
        """
        self.model.train()

        for _ in range(self.numEpochs):
            # Reset previously calculated gradients to zero.
            self.optimiser.zero_grad()
            # Predict probability of class '1
            predictions = self.model(inputs)
            # Calculate the loss of the model
            loss = self.criterion(predictions, targets)
            # Calculate the gradients
            loss.backward()
            # Update parameters with the gradient
            self.optimiser.step()

    def predict(self, inputs: torch.Tensor):
        """
        predict uses the trained model to generate the predictions.

        :param self: Description
        :param inputs: The inputs used to test the model.
        :type inputs: torch.Tensor
        """

        self.model.eval()

        with torch.no_grad():
            predictions = self.model(inputs)

        return predictions
