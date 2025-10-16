from matplotlib import axis
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

import torchvision.models as pretrainedModels

from data_path import DATA_PATH


def loadModel():
    # model = pretrainedModels.efficientnet_b7(weights="IMAGENET1K_V1")
    model = pretrainedModels.mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.eval()
    return model


def loadClasses(fileName):
    with open(DATA_PATH + fileName) as f:
        whip = {en: line.strip() for en, line in enumerate(f.readlines())}

    return whip


def loadImage(imageName: str):
    img = Image.open(DATA_PATH + imageName)

    # get evaluation metrics
    # weights = pretrainedModels.EfficientNet_B7_Weights.IMAGENET1K_V1
    weights = pretrainedModels.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    # Preprocess the image and add it to a batch.
    imgPreprocessed = preprocess(img)
    return torch.unsqueeze(imgPreprocessed, axis=0)


def generatePredictions(input, model, classes, topK=5):
    predictions = model(input)
    predictions = torch.softmax(predictions, -1)
    sortedPredictions = np.argsort(predictions.detach().numpy())[0, -topK:][::-1]

    print("\nPredictions:")
    for prediction in sortedPredictions:
        print(
            f"Class: {classes[prediction].split(',')[0]} Score: {predictions[0][prediction]*100:0.2f}%"
        )

    return predictions


img = loadImage("jellyfish.jpg")
print("Shape: ", img.shape)
print("Data type: ", img.dtype)
print("Min pixel value: ", img.min())
print("Max pixel value: ", img.max())

model = loadModel()
classes = loadClasses("imagenet_classes.txt")

generatePredictions(img, model, classes)
