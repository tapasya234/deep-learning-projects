import torch
from matplotlib import pyplot as plt
import seaborn as sn
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryConfusionMatrix,
    BinaryROC,
    BinaryAUROC,
)

from test_train_model import getPredictions


def generateConfusionMatrix(
    probabilityThreshold: float,
    predicted: torch.Tensor,
    labels: torch.Tensor,
    normalised: bool = False,
):
    if normalised:
        confMatrix = BinaryConfusionMatrix(
            threshold=probabilityThreshold, normalize="true"
        )
    else:
        confMatrix = BinaryConfusionMatrix(threshold=probabilityThreshold)

    confMatrix.update(predicted.squeeze(-1), labels)
    computedConfMatrix = confMatrix.compute().numpy()

    plt.figure(figsize=[12, 5])
    # sn.heatmap(computedConfMatrix, annot=True, fmt="d", annot_kws={"size": 16})
    sn.heatmap(computedConfMatrix, annot=True, annot_kws={"size": 16})
    plt.xlabel("Predictions")
    plt.ylabel("Targets")
    normalisedString = "Normalised" if normalised else ""
    plt.title(f"{normalisedString} Confusion Matrix ({probabilityThreshold})")
    plt.show()


def calcAccuracy(
    probabilityThreshold: float,
    predicted: torch.Tensor,
    labels: torch.Tensor,
):
    accuracy = BinaryAccuracy(threshold=probabilityThreshold)
    accuracy.update(predicted.squeeze(-1), labels)

    print(f"Accuracy ({probabilityThreshold}): {accuracy.compute():.4f}")


def calcPrecision(
    probabilityThreshold: float,
    predicted: torch.Tensor,
    labels: torch.Tensor,
):
    precision = BinaryPrecision(threshold=probabilityThreshold)
    precision.update(predicted.squeeze(-1), labels)

    print(f"Precision ({probabilityThreshold}): {precision.compute():.4f}")


def calcRecall(
    probabilityThreshold: float,
    predicted: torch.Tensor,
    labels: torch.Tensor,
):
    recall = BinaryRecall(threshold=probabilityThreshold)
    recall.update(predicted.squeeze(-1), labels)

    print(f"Recall ({probabilityThreshold}): {recall.compute():.4f}")


def calcF1Score(
    probabilityThreshold: float,
    predicted: torch.Tensor,
    labels: torch.Tensor,
):
    f1Score = BinaryF1Score(threshold=probabilityThreshold)
    f1Score.update(predicted.squeeze(-1), labels)

    print(f"F-1 Score: ({probabilityThreshold}): {f1Score.compute():.4f}")


def calcROCCurve(
    predicted: torch.Tensor,
    labels: torch.Tensor,
):
    thresholds = torch.linspace(0.001, 0.999, 1000)

    roc = BinaryROC(thresholds=thresholds)
    roc.update(predicted.squeeze(-1), labels.int())
    fpr, tpr, _ = roc.compute()

    plt.plot(fpr, tpr, label="ROC curve", color="b")

    plt.plot(
        [0, 1],
        [0, 1],
        label="Random Classifier (AUC = 0.5)",
        linestyle="--",
        lw=2,
        color="r",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.title("ROC Curve")
    plt.show()


def calcAUROC(
    predicted: torch.Tensor,
    labels: torch.Tensor,
):
    thresholds = torch.linspace(0.001, 0.999, 1000)

    auroc = BinaryAUROC(thresholds=thresholds)
    auroc.update(predicted.squeeze(-1), labels.int())
    print(f"AU-ROC Score: {auroc.compute():.4f}")


if __name__ == "__main__":
    predictions, targets = getPredictions()

    probabilityThreshold = 0.5
    generateConfusionMatrix(probabilityThreshold, predictions, targets)
    calcAccuracy(probabilityThreshold, predictions, targets)
    calcPrecision(probabilityThreshold, predictions, targets)
    calcRecall(probabilityThreshold, predictions, targets)
    calcF1Score(probabilityThreshold, predictions, targets)
    calcROCCurve(predictions, targets)
    calcAUROC(predictions, targets)
