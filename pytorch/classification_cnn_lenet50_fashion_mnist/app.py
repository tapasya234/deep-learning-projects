import os
import numpy as np
import time
import copy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader


from torchinfo import summary

import retrieve_data
import samples
from configs import DatasetConfig, TrainingConfig
import training_step
import validation_step
import lenet50_model
import plot_line_graphs
import evaluation_step


def trainModel(
    trainDataLoader: DataLoader,
    validationDataLoader: DataLoader,
    trainingConfig: TrainingConfig,
):
    model = lenet50_model.LeNet5(numClasses=dataConfig.NUM_CLASSES)
    # print(
    #     summary(
    #         model=model,
    #         input_size=(1, 1),
    #         row_settings=["var_names"],
    #     )
    # )

    model = model.float().to(trainingConfig.DEVICE)
    optimiser = optim.SGD(model.parameters(), lr=trainingConfig.LEARNING_RATE)

    bestLoss = torch.tensor(np.inf)

    # Epoch train/validation loss
    epochTrainLoss = []
    epochValidationLoss = []

    # Epoch train/validation accuracy
    epochTrainAccuracy = []
    epochValidationAccuracy = []

    # Training time measurement
    startTime = time.time()

    for epoch in range(trainingConfig.NUM_EPOCHS):
        trainLoss, trainAccuracy = training_step.train(
            DEVICE=trainingConfig.DEVICE,
            model=model,
            optimiser=optimiser,
            trainDataLoader=trainDataLoader,
            epochIndex=epoch + 1,
            numEpochs=trainingConfig.NUM_EPOCHS,
        )

        validationLoss, validationAccuracy = validation_step.validate(
            DEVICE=trainingConfig.DEVICE,
            model=model,
            validationDataLoader=validationDataLoader,
            epochIndex=epoch + 1,
            numEpochs=trainingConfig.NUM_EPOCHS,
        )

        epochTrainLoss.append(trainLoss)
        epochTrainAccuracy.append(trainAccuracy)

        epochValidationLoss.append(validationLoss)
        epochValidationAccuracy.append(validationAccuracy)

        if validationLoss < bestLoss:
            bestLoss = validationLoss
            print(f"\nModel improved. Saving model...", end="")
            bestWeights = copy.deepcopy(model.state_dict())
            torch.save(
                model.state_dict(),
                os.path.join(TrainingConfig.CHECKPOINT_DIR, "LeNet5_FashionMNIST.pth"),
            )
            print("Finished saving. \n")

        print(f"{'='*72}\n")
    print(f"Total time: {(time.time() - startTime):.2f}s\tBest loss: {bestLoss:.3f}")

    # Load model with best weights
    model.load_state_dict(bestWeights)

    plotMetrics(
        epochTrainLoss,
        epochTrainAccuracy,
        epochValidationLoss,
        epochValidationAccuracy,
        trainingConfig.NUM_EPOCHS,
    )

    return model


def plotMetrics(
    trainLoss, trainAccuracy, validationLoss, validationAccuracy, numEpochs
):
    plot_line_graphs.plot_results(
        [trainAccuracy, validationAccuracy],
        yLabel="Accuracy",
        yLim=[0.0, 1.0],
        xLim=[0, numEpochs],
        metricName=["Training Accuracy", "Validation Accuracy"],
        colour=["green", "orange"],
    )

    maxLoss = max(max(trainLoss), max(validationLoss))

    plot_line_graphs.plot_results(
        [trainLoss, validationLoss],
        yLabel="LossAccuracy",
        yLim=[0.0, 3.0],
        xLim=[0, maxLoss],
        metricName=["Training Loss", "Validation Loss"],
        colour=["green", "orange"],
    )


def createCheckpointDirectory(checkpointDir: str):
    if not os.path.exists(checkpointDir):
        os.makedirs(checkpointDir)

    print(f"Checkpoint Directory: {checkpointDir}")


if __name__ == "__main__":
    dataConfig = DatasetConfig()

    trainDataset, testDataset = retrieve_data.getData(
        dataConfig.DATA_ROOT,
        imgSize=(dataConfig.IMAGE_HEIGHT, dataConfig.IMAGE_WIDTH),
    )

    trainDataLoader, validationDataLoader, testDataLoader = (
        retrieve_data.createDataLoaders(
            trainDataset=trainDataset,
            testDataset=testDataset,
            batchSize=dataConfig.BATCH_SIZE,
            numWorkers=dataConfig.NUM_WORKERS,
        )
    )

    samples.visualiseSamples(trainDataset, predictions=None)

    trainingConfig = TrainingConfig()
    createCheckpointDirectory(trainingConfig.CHECKPOINT_DIR)

    # Train the model and save it
    trainedModel = trainModel(
        trainDataLoader=trainDataLoader,
        validationDataLoader=validationDataLoader,
        trainingConfig=trainingConfig,
    )

    # trainedModel = lenet50_model.LeNet5(dataConfig.NUM_CLASSES)
    # trainedModel.load_state_dict(
    #     torch.load(
    #         trainingConfig.CHECKPOINT_DIR + "/LeNet5_FashionMNIST.pth",
    #         weights_only=True,
    #     )
    # )

    evaluation_step.plotPredictions(
        trainedModel,
        testDataLoader,
        trainingConfig.DEVICE,
    )
