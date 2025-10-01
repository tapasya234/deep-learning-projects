from typing import Tuple

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from torch.utils.data import DataLoader

bold = f"\033[1m"
reset = f"\033[0m"


# The Training contains the following steps:
#  - Take batches of data from `trainDataLoader`.
#  - Pass the training data through the network.
#  - Compute the Cross Entropy loss using the predicted output and the training labels.
#  - To avoid gradient accumulation, remove previous gradients using `optimizer.zero_grad()`.
#  - Compute Gradients using the `backward` function.
#  - Update the weights using the `optimizer.step` function and
#    repeat until all the data is passed through the network.
# Note : While training, `nn.CrossEntropyLoss` is used which combines `nn.LogSoftMax` and `nn.NLLLoss`.
# This means that when running inference, use softmax on the raw output to convert it to probabilities.
def train(
    DEVICE: torch.device,
    model: nn.Module,
    optimiser: optim.Optimizer,
    trainDataLoader: DataLoader,
    epochIndex: int,
    shouldPrintDetails=False,
) -> Tuple[float, float]:
    model.train()
    model.to(DEVICE)

    lossStep = 0
    accuracyStep = 0
    for data, target in trainDataLoader:
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimiser.zero_grad()

        # Forward pass to the model
        output = model(data)

        loss = F.cross_entropy(output, target)
        # Find gradients w.r.t training parameters.
        loss.backward()
        # Update parameters using gradients.
        optimiser.step()
        # Convert logits to probability scores.
        probability = output.detach().softmax(dim=1)
        # Get the index of the max probability.
        probabilityIndex = probability.argmax(dim=1)

        lossStep += loss.item() * data.shape[0]
        accuracyStep += (probabilityIndex.cpu() == target.cpu()).sum()

    lossTrain = float(lossStep / len(trainDataLoader.dataset))
    accuracyTrain = float(accuracyStep / len(trainDataLoader.dataset))

    if shouldPrintDetails:
        print(f"{f'{bold}[ Epoch: {epochIndex} ]{reset}':=^80}")

        train_loss_stat = f"{bold}Train Loss: {lossTrain:.4f}{reset}"
        train_acc_stat = f"{bold}Train Acc: {accuracyTrain:.4f}{reset}"

        print(f"\n{train_loss_stat:<30}{train_acc_stat}")

    return lossTrain, accuracyTrain
