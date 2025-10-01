import time
import numpy as np
import random
from data_path import DATA_PATH

from dataclasses import dataclass
from typing import Tuple


import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchinfo import summary

from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader

import seaborn as sn
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def testSamplePredictions(
    validationsDataLoader: nn.Module, model: nn.Module, DEVICE: torch.device
):
    imagesBatch, labelsBatch = next(iter(validationsDataLoader))
    randomIndex = random.choice(range(len(imagesBatch)))

    plt.imshow(imagesBatch[randomIndex].squeeze())
    plt.title(f"Ground Truth Label: {labelsBatch[randomIndex]}", fontsize=14)
    plt.axis("off")
    plt.show()

    model.eval()

    with torch.no_grad():
        outputsBatch = model(imagesBatch.to(DEVICE))

    probabilityScoreBatch = outputsBatch.softmax(dim=1).cpu()
    probabilityScore = probabilityScoreBatch[randomIndex]
    predictionsClassIds = probabilityScore.argmax()

    for index, classProbability in enumerate(probabilityScore):
        print(f"Digit: {index} Probability: {classProbability:.3f}")

    print(f"Max Prediction Score ID: {predictionsClassIds}")


def predictBatchImages(model: nn.Module, batchInput: torch.Tensor):
    model.eval()

    batchOps = model(batchInput)
    with torch.no_grad():
        probabilities = batchOps.softmax(dim=1)

    batchClassIds = probabilities.argmax(dim=1)
    return batchClassIds.cpu()


def getPredictedLabels(
    DEVICE: torch.device,
    validationDataLoader: DataLoader,
    model: nn.Module,
):
    targetLabels = []
    predictedLabels = []

    for data, target in validationDataLoader:
        data = data.to(DEVICE)

        dataPredictions = predictBatchImages(model, data)
        predictedLabels.append(dataPredictions)
        targetLabels.append(target)

    targetLabels = torch.cat(targetLabels).numpy()
    predictedLabels = torch.cat(predictedLabels).numpy()

    return predictedLabels, targetLabels
