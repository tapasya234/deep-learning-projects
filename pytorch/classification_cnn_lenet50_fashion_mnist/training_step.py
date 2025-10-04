from typing import Tuple

from torch import device
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader


def train(
    DEVICE: device,
    model: nn.Module,
    optimiser: optim.Optimizer,
    trainDataLoader: DataLoader,
    epochIndex: int,
    numEpochs: int,
) -> Tuple[float, float]:
    model.train()

    model.to(DEVICE)

    countSamples = 0
    stepLoss = 0
    stepAccuracy = 0

    for data, target in trainDataLoader:
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimiser.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimiser.step()

        probability = F.softmax(output, dim=1)
        predictionIndex = probability.detach().argmax(dim=1)

        stepLoss += loss.item() * data.shape[0]
        stepAccuracy += (predictionIndex.cpu() == target.cpu()).sum()
        countSamples += data.shape[0]

    epochLoss = float(stepLoss / len(trainDataLoader.dataset))
    epochAccuracy = float(stepAccuracy / len(trainDataLoader.dataset))

    status = f"Train:\t Epoch: {epochIndex}/{numEpochs}"
    status += f"\tLoss: {epochLoss:.4f}"
    status += f"\tAccuracy: {epochAccuracy:.4f}"
    print(status)

    return epochLoss, epochAccuracy
