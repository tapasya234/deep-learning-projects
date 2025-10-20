import torch

from confusion_matrix import ConfusionMatrix
from test_train_model import getPredictions
from roc_curve import ROCCurve


def generateConfusionMatrix(
    probabilityThreshold: float, predictions: torch.Tensor, target: torch.Tensor
):
    confMatrix = ConfusionMatrix()
    confMatrix.reset()

    predictions = predictions > probabilityThreshold
    confMatrix.add(predictions, target)
    print(confMatrix.confusionMatrix())
    return confMatrix


def accuracy(confMatrix: ConfusionMatrix):
    truePredictions = confMatrix.TP() + confMatrix.TN()
    return (truePredictions) / (truePredictions + confMatrix.FP() + confMatrix.FN())


def precision(confMatrix: ConfusionMatrix):
    return confMatrix.TP() / (confMatrix.TP() + confMatrix.FP())


def recall(confMatrix: ConfusionMatrix):
    return confMatrix.TP() / (confMatrix.TP() + confMatrix.FN())


def f1Score(confMatrix: ConfusionMatrix):
    return (2 * confMatrix.TP()) / (
        2 * confMatrix.TP() + confMatrix.FP() + confMatrix.FN()
    )


def getConfusionMatrixEvaluations(
    probabilityThreshold: float,
    predictions: torch.Tensor,
    target: torch.Tensor,
):

    confMatrix = generateConfusionMatrix(probabilityThreshold, predictions, target)
    print(f"Confusion Matrix: \n{confMatrix.confusionMatrix()}")
    print("Accuracy: ", accuracy(confMatrix))
    print("Precision: ", precision(confMatrix))
    print("Recall: ", recall(confMatrix))
    print("F1-Score: ", f1Score(confMatrix))


def getROCEvaluations(predictions: torch.Tensor, target: torch.Tensor):
    rocAuc = ROCCurve(predictions, target)
    rocAucScore, fpr, tpr = rocAuc.getAUCScore()
    print(f"ROC AUC Score: {rocAucScore:.3f}")
    rocAuc.plotROC()


if __name__ == "__main__":
    predictions, labels = getPredictions()
    print(predictions.size())
    print(labels.size())

    probabilityThreshold = 0.5
    print(f"\n-----------\nThreshold Probability: {probabilityThreshold}")
    getConfusionMatrixEvaluations(probabilityThreshold, predictions, labels)

    probabilityThreshold = 0.7
    print(f"\n-----------\nThreshold Probability: {probabilityThreshold}")
    getConfusionMatrixEvaluations(probabilityThreshold, predictions, labels)

    print(f"\n-----------\nROC Evaluations: ")
    getROCEvaluations(predictions, labels)
