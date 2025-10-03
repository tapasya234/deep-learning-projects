import os
import time
import numpy as np
import random
from data_path import DATA_PATH

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch import optim
from torchinfo import summary

import matplotlib.pyplot as plt

from multiprocessing.spawn import freeze_support

import single_layer_perceptron as SLP
import multi_layer_perceptron as MLP
import train
import validation
import retrieve_data
import plot_line_graph
import predictions
import plot_confusion_matrix


plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["image.cmap"] = "gray"

DEVICE = torch.device("mps")
SEED_VALUE = 42
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)
torch.mps.manual_seed(SEED_VALUE)
torch.use_deterministic_algorithms(True)

# Environment variable that allows PyTorch to fall back to CPU execution
# when encountering operations that are not currently supported by MPS.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

bold = f"\033[1m"
reset = f"\033[0m"


@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES: int = 10
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 2
    DATA_ROOT: str = DATA_PATH + "/DATA_MNIST"


@dataclass(frozen=True)
class TrainingConfig:
    EPOCHS: int = 6
    LEARNING_RATE: float = 0.1


def checkSampleData(trainDataLoader: torch.utils.data.DataLoader):
    imageBatch, _ = next(iter(trainDataLoader))
    randomIndex = random.choice(range(len(imageBatch)))
    print(imageBatch.shape)
    print(randomIndex)

    plt.imshow(imageBatch[randomIndex].squeeze())
    plt.title(r"$\bf{Sample\ Example}$", fontsize=14)
    plt.show()


def trainModel(
    DEVICE: torch.device,
    model: nn.Module,
    trainDataLoader: torch.utils.data.DataLoader,
    validationDataLoader: torch.utils.data.DataLoader,
    optimiser: optim.Optimizer,
    numEpoch: int,
    shouldPrintDetails=False,
) -> Tuple:
    lossHistoryTrain = []
    lossHistoryValidation = []

    accuracyHistoryTrain = []
    accuracyHistoryValidation = []

    timeBegin = time.time()
    for epoch in range(numEpoch):
        lossTrain, accuracyTrain = train.train(
            DEVICE=DEVICE,
            model=model,
            optimiser=optimiser,
            trainDataLoader=trainDataLoader,
            epochIndex=epoch + 1,
            shouldPrintDetails=shouldPrintDetails,
        )
        lossHistoryTrain.append(lossTrain)
        accuracyHistoryTrain.append(accuracyTrain)

        lossValidation, accuracyValidation = validation.validation(
            DEVICE=DEVICE,
            model=model,
            testDataLoader=validationDataLoader,
            shouldPrintDetails=shouldPrintDetails,
        )
        lossHistoryValidation.append(lossValidation)
        accuracyHistoryValidation.append(accuracyValidation)

    print(f"Total time: {(time.time() - timeBegin):.2f}s")

    return [
        [lossHistoryTrain, lossHistoryValidation],
        [accuracyHistoryTrain, accuracyHistoryValidation],
    ]


def testModel(
    DEVICE: torch.device,
    model: nn.Module,
    trainDataLoader: torch.utils.data.DataLoader,
    validationDataLoader: torch.utils.data.DataLoader,
    numEpoch: int,
    learningRate: float,
    shouldPrintDetails=False,
):

    print(
        summary(
            model=model,
            input_size=(1, 1, 28, 28),
            row_settings=["var_names"],
        )
    )

    history = trainModel(
        DEVICE=DEVICE,
        model=model,
        trainDataLoader=trainDataLoader,
        validationDataLoader=validationDataLoader,
        optimiser=optim.SGD(model.parameters(), lr=learningRate),
        numEpoch=numEpoch,
        shouldPrintDetails=shouldPrintDetails,
    )

    plot_line_graph.plotResults(
        history[1],
        yLabel="Accuracy",
        yLim=[0.0, 1.0],
        metricName=["Training Accuracy", "Validation Accuracy"],
        colour=["green", "blue"],
    )

    maxLoss = max(max(history[0][0]), max(history[0][1]))

    plot_line_graph.plotResults(
        history[0],
        yLabel="Loss",
        yLim=[0.0, maxLoss],
        metricName=["Training Loss", "Validation Loss"],
        colour=["green", "blue"],
    )

    print(f"{"":*^50}\n\n")
    predictions.testSamplePredictions(validationDataLoader, model, DEVICE)

    predictedLabels, targetLabels = predictions.getPredictedLabels(
        DEVICE=DEVICE, validationDataLoader=validationDataLoader, model=model
    )
    accuracy = (predictedLabels == targetLabels).mean()

    print(f"\nValidations Predictions Accuracy: {accuracy * 100}%")
    plot_confusion_matrix.plotConfusionMatrix(predictedLabels, targetLabels)
    plot_confusion_matrix.plotConfusionMatrix(
        predictedLabels, targetLabels, shouldNormalise=True
    )


if __name__ == "__main__":
    # freeze_support()
    dataConfig = DatasetConfig()
    trainConfig = TrainingConfig()

    trainDataLoader, validationDataLoader = retrieve_data.getData(
        rootDir=dataConfig.DATA_ROOT,
        batchSize=dataConfig.BATCH_SIZE,
        numWorkers=dataConfig.NUM_WORKERS,
    )

    print(f"{"":-^80}\n\n\n")
    testModel(
        DEVICE=DEVICE,
        model=SLP.SingleLayerPerceptron(numClasses=dataConfig.NUM_CLASSES),
        trainDataLoader=trainDataLoader,
        validationDataLoader=validationDataLoader,
        numEpoch=trainConfig.EPOCHS,
        learningRate=trainConfig.LEARNING_RATE,
    )

    print(f"\n\n\n{"":-^80}\n\n\n")
    testModel(
        DEVICE=DEVICE,
        model=MLP.MultiLayerPerceptron(numClasses=dataConfig.NUM_CLASSES),
        trainDataLoader=trainDataLoader,
        validationDataLoader=validationDataLoader,
        numEpoch=trainConfig.EPOCHS,
        learningRate=trainConfig.LEARNING_RATE,
    )
