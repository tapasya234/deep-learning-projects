import random

import matplotlib.pyplot as plt
from torchvision import datasets

classIDToDescriptionMapping = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


def getShuffledIndices(datasetLength: int, numSamples: int):
    # random.seed(42)
    return random.sample(range(datasetLength), numSamples)


def visualiseSamples(sampleSet: datasets.FashionMNIST, predictions: None):
    plt.figure(figsize=(18, 8))

    print("Visualise Samples")
    shuffledIndices = getShuffledIndices(len(sampleSet), 10)
    for i, index in enumerate(shuffledIndices):
        plt.subplot(2, 5, i + 1)
        plt.grid(False)
        plt.imshow(sampleSet[index][0].permute(1, 2, 0), cmap="gray")
        target = sampleSet[index][1]

        if predictions is not None:
            titleColour = "green" if predictions[index] == target else "red"
            title = f"Tar: {int(target)} Pred: {int(predictions[index])}"
            plt.title(title, color=titleColour)
        else:
            plt.title(classIDToDescriptionMapping[target])
        plt.axis("off")

    plt.suptitle("Dataset samples", fontsize=18)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()
    plt.close()
