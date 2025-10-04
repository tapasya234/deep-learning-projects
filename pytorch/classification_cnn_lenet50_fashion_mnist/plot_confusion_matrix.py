import seaborn as sn
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt


def plotConfusionMatrix(predictedLabels, targetLabels, shouldNormalise=False):
    plt.figure(figsize=(15, 8))
    plt.xlabel("Predicted")
    plt.ylabel("Targets")

    if shouldNormalise:
        confusionMatrix = confusion_matrix(
            y_true=targetLabels,
            y_pred=predictedLabels,
            normalize="true",
        )
        sn.heatmap(confusionMatrix, annot=True, annot_kws={"size": 14})
        plt.title(r"$\bf{Normalized\ Confusion\ Matrix}$", color="gray")
    else:
        plt.title(r"$\bf{Confusion\ Matrix}$", color="gray")
        confusionMatrix = confusion_matrix(
            y_true=targetLabels,
            y_pred=predictedLabels,
        )
        sn.heatmap(confusionMatrix, annot=True, fmt="d", annot_kws={"size": 14})

    plt.show()
    plt.close()
