import torch
from matplotlib import pyplot as plt
import numpy as np


class ROCCurve:
    def __init__(self, predictionScores, target):
        self.predictionScores = predictionScores
        self.target = target

    def _get_fpr_tpr(self):
        thresholds = torch.linspace(0.001, 0.999, 1000).unsqueeze(1)

        # Get predictions for all thresholds
        self.predictions = self.predictionScores.T > thresholds

        truePositive, falsePositve, trueNegative, falseNegative = (
            self._get_TP_FP_TN_FN()
        )

        # Calculate True Positive Rate for all thresholds
        tpRate = truePositive.float() / (truePositive + falseNegative)

        # Calculate False Positive Rate for all thresholds
        fpRate = falsePositve.float() / (falsePositve + trueNegative)

        return fpRate.flip((0,)), tpRate.flip((0,))

    def _get_TP_FP_TN_FN(self):

        # Change datatype in bool
        self.predictions = self.predictions.bool()
        self.target = self.target.bool()

        # Calculate True Positive
        tp = (self.predictions & self.target).sum(dim=1)

        # False Positive
        fp = (self.predictions & ~self.target).sum(dim=1)

        # True Negative
        tn = (~self.predictions & ~self.target).sum(dim=1)

        # False Negative
        fn = (~self.predictions & self.target).sum(dim=1)

        return tp, fp, tn, fn

    def plotROC(self):
        # Plot TPR vs FPR
        plt.plot(*self._get_fpr_tpr(), label="ROC Curve", color="blue")
        plt.plot(
            [0, 1],
            [0, 1],
            label="Random Classifier (AUC=0.5)",
            linestyle="--",
            lw=2,
            color="red",
        )

        plt.xlabel("False Positive Rate")
        plt.xlabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.title("ROC Curve")
        plt.show()

    def getAUCScore(self):
        fpRate, tpRate = self._get_fpr_tpr()

        # Calculate the area under the curve of TPR-vs-FPR plot.
        return np.trapezoid(tpRate, fpRate), fpRate, tpRate
