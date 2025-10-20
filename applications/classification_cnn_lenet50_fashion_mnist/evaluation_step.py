import torch
from torch import device
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import samples
import plot_confusion_matrix


def evaluateModel(
    DEVICE: device,
    model: nn.Module,
    testDataLoader: DataLoader,
):
    model.eval()

    model.to(DEVICE)

    countSamples = 0
    stepLoss = 0
    stepAccuracy = 0

    for data, target in testDataLoader:
        data, target = data.to(DEVICE), target.to(DEVICE)

        with torch.no_grad():
            output = model(data)
        loss = F.cross_entropy(output, target).item()

        probability = F.softmax(output, dim=1)
        predictionIndex = probability.detach().argmax(dim=1)

        stepLoss += loss * data.shape[0]
        stepAccuracy += (predictionIndex.cpu() == target.cpu()).sum()
        countSamples += data.shape[0]

    print("\n")

    print(
        f"Evaluation Loss: {float(stepLoss/countSamples):.4f} Accuracy: {float(stepAccuracy/countSamples):.4f}"
    )

    return


def getBatchPredictions(model: nn.Module, batchInputs: torch.tensor):
    model.eval()

    with torch.no_grad():
        batchLogits = model(batchInputs)

    batchClassIds = batchLogits.argmax(dim=1)
    return batchClassIds.cpu()


def plotPredictions(
    model: nn.Module,
    testDataLoader: DataLoader,
    DEVICE: str = "cpu",
):
    model.to(DEVICE)

    print("Evaluating model...")
    evaluateModel(DEVICE=DEVICE, model=model, testDataLoader=testDataLoader)

    targetsBatch = []
    predictionsBatch = []

    for data, target in testDataLoader:
        data = data.to(DEVICE)

        predictionsBatch.append(getBatchPredictions(model, data))
        targetsBatch.append(target)

    targetsBatch = torch.cat(targetsBatch).numpy()
    predictionsBatch = torch.cat(predictionsBatch).numpy()
    accuracy = (predictionsBatch == targetsBatch).mean()
    print(f"Evaluate Predictions Accuracy: {accuracy * 100}%")

    plot_confusion_matrix.plotConfusionMatrix(
        predictionsBatch,
        targetsBatch,
        shouldNormalise=True,
    )

    samples.visualiseSamples(testDataLoader.dataset, predictionsBatch)

    return
