import configs
from model import MyModel
from train_model import train
from validate_model import validate

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader


def setup_system() -> None:
    torch.manual_seed(configs.SystemConfig.SEED)
    torch.mps.manual_seed(configs.SystemConfig.SEED)


def optimizeModel(
    numClasses: int,
    trainingConfig: configs.TrainingConfig,
    trainDataLoader: DataLoader,
    testDataLoader: DataLoader,
):

    setup_system()

    # initiate model
    model = MyModel(numClasses=numClasses)

    # send model to device (GPU/CPU)
    model.to(trainingConfig.DEVICE)

    # optimizer
    optimizer = SGD(model.parameters(), lr=trainingConfig.LEARNING_RATE)

    best_loss = torch.tensor(np.inf)
    best_accuracy = torch.tensor(0)

    # epoch train/test loss
    epoch_train_loss = np.array([])
    epoch_test_loss = np.array([])

    # epch train/test accuracy
    epoch_train_acc = np.array([])
    epoch_test_acc = np.array([])

    # trainig time measurement
    t_begin = time.time()
    for epoch in range(trainingConfig.NUM_EPOCHS):

        train_loss, train_acc = train(
            trainingConfig.DEVICE,
            model,
            optimizer,
            trainDataLoader,
            epoch,
            trainingConfig.LOG_INTERVAL,
        )

        epoch_train_loss = np.append(epoch_train_loss, [train_loss])

        epoch_train_acc = np.append(epoch_train_acc, [train_acc])

        elapsed_time = time.time() - t_begin
        speed_epoch = elapsed_time / (epoch + 1)
        speed_batch = speed_epoch / len(trainDataLoader)
        eta = speed_epoch * trainingConfig.NUM_EPOCHS - elapsed_time

        print(
            "Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
                elapsed_time, speed_epoch, speed_batch, eta
            )
        )

        if epoch % trainingConfig.TEST_INTERVAL == 0:
            current_loss, current_accuracy = validate(
                trainingConfig.DEVICE, model, testDataLoader
            )

            epoch_test_loss = np.append(epoch_test_loss, [current_loss])

            epoch_test_acc = np.append(epoch_test_acc, [current_accuracy])

            if current_loss < best_loss:
                best_loss = current_loss

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                print("Accuracy improved, saving the model.\n")
                save_model(
                    model,
                    trainingConfig.DEVICE,
                    model_dir=trainingConfig.MODELS_DIR,
                    model_file_name=trainingConfig.MODEL_NAME,
                )

    print(
        "Total time: {:.2f}, Best Loss: {:.3f}, Best Accuracy: {:.3f}".format(
            time.time() - t_begin, best_loss, best_accuracy
        )
    )

    return model, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc


def save_model(
    model: nn.Module,
    device: torch.device,
    model_dir="models",
    model_file_name="cifar10_cnn_model.pt",
):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_file_name)

    # make sure you transfer the model to the correct format.
    model.to(device)

    # save the state_dict
    if int(torch.__version__.split(".")[1]) >= 6:
        torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)

    else:
        torch.save(model.state_dict(), model_path)

    return
