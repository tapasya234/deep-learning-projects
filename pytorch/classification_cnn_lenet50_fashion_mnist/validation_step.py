from typing import Tuple

import torch
from torch import device
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


def validate(
    DEVICE: device,
    model: nn.Module,
    validationDataLoader: DataLoader,
    epochIndex: int,
    numEpochs: int,
) -> Tuple[float, float]:
    model.eval()
    model.to(DEVICE)

    countSamples = 0
    stepLoss = 0
    stepAccuracy = 0

    for data, target in validationDataLoader:
        data, target = data.to(DEVICE), target.to(DEVICE)

        with torch.no_grad():
            output = model(data)
        loss = F.cross_entropy(output, target).item()

        probability = F.softmax(output, dim=1)
        predictionIndex = probability.detach().argmax(dim=1)

        stepLoss += loss * data.shape[0]
        stepAccuracy += (predictionIndex.cpu() == target.cpu()).sum()
        countSamples += data.shape[0]

    epochLoss = float(stepLoss / len(validationDataLoader.dataset))
    epochAccuracy = float(stepAccuracy / len(validationDataLoader.dataset))

    status = f"Validation:\t Epoch: {epochIndex}/{numEpochs}"
    status += f"\tLoss: {epochLoss:.4f}"
    status += f"\tAccuracy: {epochAccuracy:.4f}"
    print(status)

    return epochLoss, epochAccuracy
