import random
import time
import math

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import distributions as dist
from torchinfo import summary

import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt
from matplotlib import animation

plt.rcParams["figure.figsize"] = (15, 6)
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14

block_plot = False

DEVICE = "mps"
BOLD = f"\033[1m"
RESET = f"\033[0m"

SEED_VALUE = 42
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)


def generateData(
    meanClass0=[4.0, 20.0],
    stdDevClass0=[1.0, 1.0],
    meanClass1=[5.5, 23.0],
    stdDevClass1=[0.6, 0.8],
    numPtsClass0=200,
    numPtsClass1=200,
):
    """
    generateData will generate some synthetic data to represent
    two features from each of the two classes.
    """
    distClass0 = dist.Normal(
        loc=torch.tensor(meanClass0), scale=torch.tensor(stdDevClass0)
    )
    pointsClass0 = distClass0.sample((numPtsClass0,))

    distClass1 = dist.Normal(
        loc=torch.tensor(meanClass1), scale=torch.tensor(stdDevClass1)
    )
    pointsClass1 = distClass1.sample((numPtsClass1,))
    return pointsClass0, pointsClass1


def prepareData(pointsClass0, pointsClass1):
    """
    prepareData will combine the features and labels from each class
    into a single data array and labels array.
    """

    labels = torch.cat(
        [
            torch.zeros(pointsClass0.shape[0], dtype=torch.float),
            torch.ones(pointsClass1.shape[0], dtype=torch.float),
        ],
        dim=0,
    ).unsqueeze(dim=1)

    dataPoints = torch.cat([pointsClass0, pointsClass1], dim=0)

    print(f"Datapoints size: {dataPoints.shape}")
    print(f"Labels size: {labels.shape}")
    return dataPoints, labels


def visualiseDataset(pointsClass0, pointsClass1):
    """
    visualiseDataset will generate a scatter plot of the points from both classes.
    """
    plt.figure(figsize=(20, 10))
    plt.scatter(
        pointsClass0[:, 0],
        pointsClass0[:, 1],
        color="orange",
        alpha=0.5,
        label="Class: 0",
    )
    plt.scatter(
        pointsClass1[:, 0],
        pointsClass1[:, 1],
        color="green",
        alpha=0.5,
        label="Class: 1",
    )
    plt.legend()
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.xlim([0, 10])
    plt.ylim([16, 28])
    plt.grid(True)

    # plt.show(block=block_plot)
    plt.show()
    plt.close()


def normaliseData(
    data: torch.Tensor,
    mean: torch.Tensor,
    stdDev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    normaliseData will normalise the data using mean and standard deviation of the data.

    :param data: Data that will be normalised
    :type data: torch.Tensor
    :param mean: Mean value to be used for normalising data
    :type mean: torch.Tensor
    :param stdDev: Standard Deviation value to be used for normalising data
    :type stdDev: torch.Tensor
    """
    print("Normalise Data")
    print(f"Shape: {data.shape} Mean: {mean} StdDev: {stdDev}")
    return (data - mean) / stdDev


def trainModel(
    model: nn.Module,
    dataPoints: torch.Tensor,
    labels: torch.Tensor,
    epochs,
    optimiser: optim.SGD,
    batchSize=10,
):
    numBatches = math.ceil(len(labels) / batchSize)

    lossHistory = []
    accuracyHistory = []

    model = model.to(DEVICE)
    model.train()

    timeTrainingBegin = time.time()  # Used to measure how long the training takes.
    for epochIndex in range(epochs):
        # Shuffle data to the start of each epoch.
        shuffledIndices = torch.randperm(len(labels))
        shuffledData = dataPoints[shuffledIndices]
        shuffledLabels = labels[shuffledIndices]

        lossStep = 0
        accuracyStep = 0

        for batchIndex in range(numBatches):
            startIndex = batchIndex * batchSize
            endIndex = startIndex + batchSize

            batchedData = shuffledData[startIndex:endIndex].to(DEVICE)
            batchedTargets = shuffledLabels[startIndex:endIndex].to(DEVICE)

            # Set the weight gradients to zero for every mini-batch to
            # avoid gradient accumulation.
            optimiser.zero_grad()

            logits = model(batchedData)
            loss = F.binary_cross_entropy_with_logits(logits, batchedTargets)

            # Compute fradients with back propagation.
            loss.backward()

            # Update model weights
            optimiser.step()

            # Convert the output logits to probabilities.
            predictions = logits.sigmoid()

            # Batch loss
            lossStep += loss.item() * batchedData.shape[0]

            # Batch Accuracy
            accuracyStep += (
                (predictions > 0.5).int().cpu() == batchedTargets.cpu()
            ).sum()

        lossEpoch = float(lossStep / len(labels))
        lossHistory.append(lossEpoch)
        accuracyEpoch = float(accuracyStep / len(labels))
        accuracyHistory.append(accuracyEpoch)

        if (epochIndex + 1) % 15 == 0:
            print(f"{f'{BOLD}[ Epoch: {epochIndex+1} ]{RESET}':=^80}")
            lossTrainStat = f"{BOLD}Loss: {lossEpoch:.4f}{RESET}"
            accuracyTrainStat = f"{BOLD}Accuracy: {accuracyEpoch:.4f}{RESET}"

            print(f"\n{lossTrainStat:<30}{accuracyTrainStat}")

            print(f"{'='*72}\n")

    print(f"Total time: {(time.time()-timeTrainingBegin):.2f}s")
    return model, lossHistory, accuracyHistory


def plotTrainingResults(dataPoints, yLabel, title):
    plt.figure(figsize=(20, 6))
    plt.plot(range(len(dataPoints)), dataPoints)
    plt.xlabel("Epochs")
    plt.ylabel(yLabel)
    plt.title(title)
    plt.grid(True)
    plt.xlim([0, 100])
    plt.ylim([0, 1])

    plt.show()
    plt.close()


def getPredictedClass(prediction):
    """
    getPredictedClass maps the predicted score to the appropriate class name.

    :param prediction: prediction score to be mapped
    """

    prediction = prediction.squeeze().numpy()
    if prediction > 0.5:
        return "Werecat"
    else:
        return "Werewolf"


def getPredictionScore(model: nn.Module, inputs, device="cpu"):
    """
    predictClass converts output logits to probability scores.

    :param model: Trained model which is used to predict the test data.
    :type model: nn.Module
    :param inputs: Test data to be predicted.
    :param device: Device on which the algorithm is being processed.
    """
    model.eval()

    with torch.no_grad():
        logits = model(inputs.to(device)).cpu()

    return logits.sigmoid()


def predictData(
    model: nn.Module,
    testData: torch.Tensor,
    mean: torch.Tensor,
    stdDev: torch.Tensor,
):
    """
    predictData normalises the test data, used the trained model to
    predict the data and returns the associated class label.

    :param model: Trained model which is used to predict the test data.
    :type model: nn.Module
    :param testData: Test data to be predicted.
    :type testData: torch.Tensor
    :param mean: Mean value to be used for normalising data.
    :type mean: torch.Tensor
    :param stdDev: Standard Deviation value to be used for normalising data.
    :type stdDev: torch.Tensor
    """
    normalisedData = normaliseData(testData, mean=mean, stdDev=stdDev)
    predictionScore = getPredictionScore(model, normalisedData, device=DEVICE)
    return getPredictedClass(predictionScore)


def wx_plus_b(W, X, B):
    """
    Neuron: WX + B
    """
    return torch.matmul(X, W) + B


def sigmoid(z):
    """
    Sigmoid Activation
    """
    return 1 / (1 + torch.exp(-z))


def retrieveWeightsAndBias(
    model: nn.Module,
    mean: torch.Tensor,
    stdDev: torch.Tensor,
):
    weights = model.weight.detach().cpu()
    bias = model.bias.detach().cpu()

    w1 = weights[0][0]
    w2 = weights[0][1]
    b = bias[0]

    print("Weights associated with normalised data")
    print("Bias: ", b)
    print("Weight1: ", w1)
    print("Weight2: ", w2)

    # Unnormalise the weights for use in diagnostics.
    w1 = w1 / stdDev[0]
    w2 = w2 / stdDev[1]
    b = b - w1 * mean[0] - w2 * mean[1]

    print("Weights associated with unnormalised data")
    print("Bias: ", b.numpy())
    print("Weight1: ", w1.numpy())
    print("Weight2: ", w2.numpy())
    return w1, w2, b


def plotSigmoid(pointsClass0, pointsClass1, w1, w2, b):
    W = torch.zeros((2, 1))
    W[0][0] = w1
    W[1][0] = w2

    # Compute sigmoid activations for Class 0
    zClass0 = wx_plus_b(W, pointsClass0, b)
    yPredictedClass0 = sigmoid(zClass0)

    # Compute sigmoid activations for Class 1
    zClass1 = wx_plus_b(W, pointsClass1, b)
    yPredictedClass1 = sigmoid(zClass1)

    plt.figure(figsize=(20, 7))
    plt.scatter(
        zClass0, yPredictedClass0, s=20, alpha=0.5, color="orange", label="Class: 0"
    )
    plt.scatter(
        zClass1, yPredictedClass1, s=20, alpha=0.5, color="green", label="Class: 1"
    )
    plt.xlabel("Z")
    plt.ylabel("Sigmoid(Z)")
    plt.plot([0, 0], [0, 1], color="blue")
    plt.plot([-10, 10], [0.5, 0.5], color="red")
    plt.xlim([-10, 10])
    plt.ylim([0, 1])
    plt.legend()
    plt.show()
    plt.close()


def plotDecisionBoundary(pointsClass0, pointsClass1, w1, w2, b):
    plt.figure(figsize=(20, 8))
    plt.scatter(
        pointsClass0[:, 0],
        pointsClass0[:, 1],
        color="orange",
        alpha=0.5,
        label="Class: 0",
    )
    plt.scatter(
        pointsClass1[:, 0],
        pointsClass1[:, 1],
        color="green",
        alpha=0.5,
        label="Class: 1",
    )

    x1 = torch.linspace(0.0, 10.0, 1000)
    x2 = -(w1 / w2) * x1 - b / w2

    plt.plot(x1, x2, c="black", alpha=0.5)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.xlim([0, 10])
    plt.ylim([16, 28])
    plt.grid(True)
    plt.legend()

    plt.show()
    plt.close()


# - True  Positive (class 1 is classified as class 1)
# - False Positive (class 0 is classified as class 1)
# - True  Negative (class 0 is classified as class 0)
# - False Negative (class 1 is classified as class 0)
def plotConfusionMatrix(
    trainedModel: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor
):
    predictions = getPredictionScore(trainedModel, X_train, device=DEVICE)
    predictedLabels = torch.where(predictions > 0.5, 1, 0)

    confusionMatrix = confusion_matrix(
        y_train.squeeze().numpy(), predictedLabels.numpy()
    )

    plt.figure(figsize=[12, 5])
    sn.heatmap(confusionMatrix, annot=True, fmt="d", annot_kws={"size": 16})
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.show()
    plt.close()


pointsClass0, pointsClass1 = generateData()
visualiseDataset(pointsClass0, pointsClass1)
X_train, y_train = prepareData(pointsClass0, pointsClass1)

mean = X_train.mean(0)
stdDev = X_train.std(0)
X_train = normaliseData(X_train, mean=mean, stdDev=stdDev)

model = nn.Linear(in_features=2, out_features=1)
print(summary(model, input_size=(1, 2)))
optimiserSGD = optim.SGD(model.parameters(), lr=0.1)

trainedModel, lossHistory, accuracyHistory = trainModel(
    model=model,
    dataPoints=X_train,
    labels=y_train,
    epochs=100,
    optimiser=optimiserSGD,
    batchSize=16,
)
plotTrainingResults(lossHistory, "Loss", "Training Loss")
plotTrainingResults(accuracyHistory, "Classification Accuracy", "Training Accuracy")

testData = torch.tensor([5.0, 21.0]).unsqueeze(0)
print(
    f"Data: {testData} Class: {predictData(model=trainedModel, testData=testData, mean=mean, stdDev=stdDev)}"
)

testData = torch.tensor([5.0, 22.0]).unsqueeze(0)
print(
    f"Data: {testData} Class: {predictData(model=trainedModel, testData=testData, mean=mean, stdDev=stdDev)}"
)

w1, w2, b = retrieveWeightsAndBias(trainedModel, mean, stdDev)
plotSigmoid(pointsClass0, pointsClass1, w1, w2, b)
plotDecisionBoundary(pointsClass0, pointsClass1, w1, w2, b)
plotConfusionMatrix(trainedModel, X_train, y_train)
