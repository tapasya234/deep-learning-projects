import os
import requests
import zipfile
import glob
import ast

from typing import List

from PIL import Image, ImageFile

import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision.models import resnet50, ResNet50_Weights

from data_path import DATA_PATH


def downloadFile(url, saveName):
    if not os.path.exists(saveName):
        file = requests.get(url)
        open(saveName, "wb").write(file.content)


def unzipFile(zipFile=None):
    try:
        with zipfile.ZipFile(zipFile) as z:
            z.extractall("./")
            print("Extracted all")
    except Exception as e:
        print(type(e))
        print(e.args)
        print(e)
        print("Invalid File")


def getRequiredFiles():
    downloadFile(
        "https://raw.githubusercontent.com/Lasagne/Recipes/master/examples/resnet50/imagenet_classes.txt",
        DATA_PATH + "imagenet_classes.txt",
    )

    downloadFile(
        "https://www.dropbox.com/s/cprmbjb8l3olyiv/dataset_image_classification_cnn.zip?dl=1",
        DATA_PATH + "dataset_image_classification_cnn.zip",
    )

    unzipFile(zipFile=DATA_PATH + "dataset_image_classification_cnn.zip")


def getImagePaths() -> List:
    return glob.glob(DATA_PATH + "dataset_image_classification_cnn/*.jpg")


def loadModel():
    model = resnet50(weights="IMAGENET1K_V1")
    model.eval()
    return model


def classifyImages(imagePaths, model):
    with open(DATA_PATH + "imagenet_classes.txt") as f:
        whip = {en: line.strip() for en, line in enumerate(f.readlines())}

    # get evaluation transformations
    weights = ResNet50_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    for _, imagePath in enumerate(imagePaths):
        image = Image.open(imagePath)
        imgTransformed = preprocess(image)

        imgBatch = torch.unsqueeze(imgTransformed, axis=0)
        predictions = model(imgBatch)
        predictions = torch.softmax(predictions, -1)

        sortedPredictions = np.argsort(predictions.detach().numpy())[0, -5:][::-1]

        # Display the image
        print("Path: ", imagePath)
        for num, pred in enumerate(sortedPredictions):
            print(
                f"Prediction: {num}: {whip[pred].split(',')[0]}, {predictions[0][pred]*100:.2f}%"
            )
        print("\n\n")

        imagePlt = plt.imread(imagePath)
        plt.imshow(imagePlt)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    getRequiredFiles()
    model = loadModel()
    imagePaths = getImagePaths()
    classifyImages(imagePaths, model)
