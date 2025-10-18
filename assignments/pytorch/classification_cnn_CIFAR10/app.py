import configs
from data_retrieval import getData
from model import MyModel
from optimise_model import optimizeModel


import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
import torchvision

from torchinfo import summary


def loadModel(
    numClasses: int,
    trainingConfig: configs.TrainingConfig,
    trainDataLoader: DataLoader,
    testDataLoader: DataLoader,
):
    # initialize the model
    cnn_model = MyModel(numClasses=numClasses)
    print(summary(cnn_model))

    if not os.path.exists(trainingConfig.MODELS_DIR):
        os.makedirs(trainingConfig.MODELS_DIR)

    model_path = os.path.join(trainingConfig.MODELS_DIR, trainingConfig.MODEL_NAME)

    if not os.path.exists(model_path):
        optimizeModel(
            numClasses,
            trainingConfig,
            trainDataLoader,
            testDataLoader,
        )

    # loading the model and getting model parameters by using load_state_dict
    cnn_model.load_state_dict(torch.load(model_path))

    return cnn_model


if __name__ == "__main__":
    dataConfig = configs.DatasetConfig()
    trainDataLoader, testDataLoader = getData(
        batchSize=dataConfig.BATCH_SIZE,
        dataRoot=dataConfig.DATA_ROOT,
        imgWidth=dataConfig.IMAGE_WIDTH,
        imgHeight=dataConfig.IMAGE_HEIGHT,
        numWorkers=dataConfig.NUM_WORKERS,
    )

    savedModel = loadModel(
        numClasses=dataConfig.NUM_CLASSES,
        trainingConfig=configs.TrainingConfig(),
        trainDataLoader=trainDataLoader,
        testDataLoader=testDataLoader,
    )
