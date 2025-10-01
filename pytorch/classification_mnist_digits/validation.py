from typing import Tuple

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from torch.utils.data import DataLoader

bold = f"\033[1m"
reset = f"\033[0m"


def validation(
    DEVICE: torch.device,
    model: nn.Module,
    testDataLoader: DataLoader,
    shouldPrintDetails: False,
) -> Tuple[float, float]:
    model.eval()
    model.to(DEVICE)

    lossStep = 0
    accuracyStep = 0
    for data, target in testDataLoader:
        data, target = data.to(DEVICE), target.to(DEVICE)

        with torch.no_grad():
            output = model(data)

        lossTest = F.cross_entropy(output, target).item()

        prob = output.softmax(dim=1)
        predIndex = prob.argmax(dim=1)

        lossStep += lossTest * data.shape[0]
        accuracyStep += (predIndex.cpu() == target.cpu()).sum()

    lossValidation = float(lossStep / len(testDataLoader.dataset))
    accuracyValidation = float(accuracyStep / len(testDataLoader.dataset))

    if shouldPrintDetails:
        valid_loss_stat = f"{bold}Valid Loss: {lossValidation:.4f}{reset}"
        valid_acc_stat = f"{bold}Valid Acc: {accuracyValidation:.4f}{reset}"

        print(f"\n{valid_loss_stat:<30}{valid_acc_stat}")

    return lossValidation, accuracyValidation
