from cProfile import label
import random
import math
import os
import math
from turtle import forward
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchinfo import summary

# Set plotting related parameters.
plt.rcParams["figure.figsize"] = (15, 6)
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 12

warnings.filterwarnings(action="ignore", category=UserWarning)

BATCH_SIZE = 1


def systemConfig(SEED_VALUE=42, package_list=None):
    """
    systemConfig configures the system environment for PyTorch based operations.

    :param SEED_VALUE: Seed Value for random number generation. Default value is 42.
    :param package_list: String containing a lit of additional packages to install.
    Typically used when running in Google Colab or Kaggle.

    :return tuple: A tuple containing device name as a string and
    boolean indicating GPU availability.
    """

    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)

    if torch.cuda.is_available():
        print("Using CUDA GPU")

        # Set the device to the first CUDA device.
        DEVICE = torch.device("cuda")
        GPU_AVAILABLE = True

        torch.cuda.manual_seed(SEED_VALUE)
        torch.cuda.manual_seed_all(SEED_VALUE)

        # Performance and deterministic behavior.
        torch.backends.cudnn.enabled = (
            True  # Provides highly optimized primitives for DL operations.
        )
        torch.backends.cudnn.deterministic = (
            True  # Insures deterministic even when above cudnn is enabled.
        )
        torch.backends.cudnn.benchmark = (
            False  # Setting to True can cause non-deterministic behavior.
        )

    elif torch.backends.mps.is_available() or torch.backends.mps.is_built():
        print("using Apple Silicon GPU")

        # Set the device to Apple Silicon GPU Metal Performance Shader (MPS).
        DEVICE = torch.device("mps")

        # Environment variable that allows PyTorch to fall back to CPU execution
        # when encountering operations that are not currently supported by MPS.
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        GPU_AVAILABLE = True

        torch.mps.manual_seed(SEED_VALUE)
        torch.use_deterministic_algorithms(True)
    else:
        print("Using CPU")

        DEVICE = torch.device("cpu")
        GPU_AVAILABLE = False

        torch.use_deterministic_algorithms(True)

    print("Device: ", DEVICE)
    print("Is GPU Available: ", GPU_AVAILABLE)
    return DEVICE, GPU_AVAILABLE


# DEVICE, GPU_AVAILABLE = systemConfig(SEED_VALUE=21)

autoMPG = fetch_ucirepo(id=9)
columnsNames = autoMPG.variables.name
# print(columnsNames.shape)

dataset = pd.concat([autoMPG.data.features, autoMPG.data.targets], axis=1)
# print(dataset.columns)

dataset = dataset.dropna()

trainDataset = dataset.sample(frac=0.8, random_state=28)
testDataset = dataset.drop(trainDataset.index)

# print(trainDataset.shape)
# print(testDataset.shape)

X_train = trainDataset.copy()
X_test = testDataset.copy()

y_train = X_train.pop("mpg")
y_test = X_test.pop("mpg")

# Normalize 'horsepower' feature
mean_hp = X_train["horsepower"].mean()
std_hp = X_train["horsepower"].std()
X_train["horsepower_scaled"] = (X_train["horsepower"] - mean_hp) / std_hp
X_test["horsepower_scaled"] = (X_test["horsepower"] - mean_hp) / std_hp

# print("horsepower (Before) Mean: {} Std: {}", mean_hp, std_hp)
# print(
#     "horsepower (After) Mean: {} Std: {}",
#     X_train["horsepower_scaled"].mean(),
#     X_train["horsepower_scaled"].std(),
# )

# Normalize 'displacement' feature
mean_disp = X_train["displacement"].mean()
std_disp = X_train["displacement"].std()
X_train["displacement_scaled"] = (X_train["displacement"] - mean_disp) / std_disp
X_test["displacement_scaled"] = (X_test["displacement"] - mean_disp) / std_disp

# print("displacement (Before) Mean: {} Std: {}", mean_disp, std_disp)
# print(
#     "displacement (After) Mean: {} Std: {}",
#     X_train["displacement_scaled"].mean(),
#     X_train["displacement_scaled"].std(),
# )

#  Split the training set into 70% training and 30% validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.3, random_state=0
)
y_train_torch = torch.from_numpy(y_train.values).reshape(-1, 1).to(torch.float32)
y_valid_torch = torch.from_numpy(y_valid.values).reshape(-1, 1).to(torch.float32)
y_test_torch = torch.from_numpy(y_test.values).reshape(-1, 1).to(torch.float32)

X_train_hp = (
    torch.from_numpy(X_train["horsepower_scaled"].values)
    .reshape(-1, 1)
    .to(torch.float32)
)

X_valid_hp = (
    torch.from_numpy(X_valid["horsepower_scaled"].values)
    .reshape(-1, 1)
    .to(torch.float32)
)
X_test_hp = (
    torch.from_numpy(X_test["horsepower_scaled"].values)
    .reshape(-1, 1)
    .to(torch.float32)
)

X_train_hp_dp = (
    torch.from_numpy(X_train[["horsepower_scaled", "displacement_scaled"]].values)
    .reshape(-1, 2)
    .to(torch.float32)
)
X_valid_hp_dp = (
    torch.from_numpy(X_valid[["horsepower_scaled", "displacement_scaled"]].values)
    .reshape(-1, 2)
    .to(torch.float32)
)
X_test_hp_dp = (
    torch.from_numpy(X_test[["horsepower_scaled", "displacement_scaled"]].values)
    .reshape(-1, 2)
    .to(torch.float32)
)


def plotLoss(lossCurveTrain, lossCurveEval):
    plt.figure(figsize=(15, 5))
    plt.plot(lossCurveTrain, label="Train Loss")
    plt.plot(lossCurveEval, label="Valid Loss")
    plt.ylim([0, 30])
    plt.xlabel("Epoch")
    plt.ylabel("Error MPG")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()
    return


def plot_horsepower(x, y):
    plt.figure(figsize=(15, 5))

    plt.scatter(
        (X_train["horsepower_scaled"] * std_hp) + mean_hp,
        y_train,
        label="Data",
        color="green",
        alpha=0.5,
    )
    plt.plot(x, y, color="k", label="Predictions")
    plt.xlabel("Horsepower")
    plt.ylabel("MPG")
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.close()


def plot_horsepowerDisplacement(x_surf, y_surf, yPredicted):
    fig = plt.figure(figsize=(20, 10))

    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(
        (X_train["horsepower_scaled"] * std_hp) + mean_hp,
        (X_train["horsepower_scaled"] * std_disp) + mean_disp,
        y_train,
        c="blue",
        marker="o",
        alpha=0.5,
    )
    ax.plot_surface(
        x_surf,
        y_surf,
        yPredicted.reshape(x_surf.shape),
        color="yellow",
        alpha=0.5,
    )
    ax.set_xlabel("Horsepower")
    ax.set_ylabel("Displacement")
    ax.set_zlabel("MPG")
    ax.view_init(9, -40)

    ax = fig.add_subplot(122, projection="3d")
    ax.scatter(
        (X_train["horsepower_scaled"] * std_hp) + mean_hp,
        (X_train["horsepower_scaled"] * std_disp) + mean_disp,
        y_train,
        c="blue",
        marker="o",
        alpha=0.5,
    )
    ax.plot_surface(
        x_surf,
        y_surf,
        yPredicted.reshape(x_surf.shape),
        color="yellow",
        alpha=0.5,
    )
    ax.set_xlabel("Horsepower")
    ax.set_ylabel("Displacement")
    ax.set_zlabel("MPG")
    ax.view_init(9, 140)

    plt.show()
    plt.close()


def generateHistogram(lossCurveTrain, lossCurveEval):
    print("Histogram")
    hist = pd.DataFrame({"train_loss": lossCurveTrain, "valid_loss": lossCurveEval})
    print(hist.describe())


def trainModel(
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    dataset: tuple,
):
    data, target = dataset
    model.train()  # Set model in training mode

    outputs = model(data)  # Perform forward pass through the model
    loss = loss_fn(outputs, target)  # Calculate La loss on the model predictions.
    optimiser.zero_grad()  # Reset gradients
    loss.backward()  # Calculate gradients based on loss
    optimiser.step()  # Update parameters

    return loss.detach().item()


def evaluateModel(model: torch.nn.Module, loss_fn: torch.nn.Module, dataset: tuple):
    data, target = dataset
    model.eval()  # Set model in evaluation mode
    with torch.no_grad():
        outputs = model(data)

    loss = loss_fn(outputs, target)
    return loss.item()


@torch.inference_mode()
def generatePredictions(model: torch.nn.Module, data):
    model.eval()
    return model(data)


class Regressor_Linear(nn.Module):
    # Initialize the parameter
    def __init__(self, inFeatures=1, outFeatures=1):
        super().__init__()

        # Define single Linear layer
        self.linear_1 = nn.Linear(in_features=inFeatures, out_features=outFeatures)

    def forward(self, x):
        return self.linear_1(x)


class Regressor_DNN(nn.Module):
    def __init__(self, inFeatures=2, outFeatures=1, intermediate=10):
        super().__init__()

        self.linear_1 = nn.Linear(in_features=inFeatures, out_features=intermediate)
        self.linear_2 = nn.Linear(in_features=intermediate, out_features=intermediate)
        self.linear_3 = nn.Linear(in_features=intermediate, out_features=outFeatures)

    def forward(self, x):
        # First Linear Layer --> ReLU activation
        pred = F.relu(self.linear_1(x))

        # Second Linear Layer --> ReLU activation
        pred = F.relu(self.linear_2(pred))

        # Third Linear Layer
        pred = self.linear_3(pred)
        return pred


def getModelSummary(model: torch.nn.Module, inputFeatures):
    summary(
        model,
        input_size=(
            BATCH_SIZE,
            inputFeatures,
        ),
        # device=DEVICE,
        device="cpu",
        col_names=("input_size", "output_size", "num_params"),
    )


# Initialize the Loss function.
# L1Loss is being used to measure the MAE between the input x and target y.
criterion = torch.nn.L1Loss()


def linearModel_1D(X_train, y_train_torch, y_valid_torch, shouldDisplayPlot=False):
    model = Regressor_Linear(inFeatures=1, outFeatures=1)
    getModelSummary(model, 1)

    # Initialize Optimiser by passing the parameters of the model and the learning rate to use.
    # The model parameters passed to the optimizer will be updated during training.
    optimiser = optim.Adam(model.parameters(), lr=0.1)

    # Record the training and validation loss metrics
    lossCurve_train = []
    lossCurve_eval = []

    for epoch in range(500):
        lossTrain = trainModel(model, optimiser, criterion, (X_train_hp, y_train_torch))
        lossCurve_train.append(lossTrain)

        lossEval = evaluateModel(model, criterion, (X_valid_hp, y_valid_torch))
        lossCurve_eval.append(lossEval)

    if shouldDisplayPlot:
        plotLoss(lossCurve_train, lossCurve_eval)
    # generateHistogram(lossCurve_train, lossCurve_eval)

    #  Model Prediction
    x = torch.linspace(
        X_train["horsepower"].min(),
        X_train["horsepower"].max(),
        len(X_train["horsepower"]),
    )
    yPredicted = generatePredictions(
        model, data=((x.view(-1, 1) - mean_hp) / std_hp)
    ).numpy()

    if shouldDisplayPlot:
        plot_horsepower(x, yPredicted)

    with torch.no_grad():
        testResults = criterion(model(X_test_hp), y_test_torch).numpy()
    return lossCurve_train, lossCurve_eval, testResults


def linearModel_2D(X_train, y_train_torch, y_valid_torch, shouldDisplayPlot=False):
    model = Regressor_Linear(inFeatures=2, outFeatures=1)
    getModelSummary(model, 2)

    optimiser = optim.Adam(model.parameters(), lr=0.1)

    lossCurve_train = []
    lossCurve_eval = []

    for epoch in range(500):
        lossTrain = trainModel(
            model, optimiser, criterion, (X_train_hp_dp, y_train_torch)
        )
        lossCurve_train.append(lossTrain)

        lossEval = evaluateModel(model, criterion, (X_valid_hp_dp, y_valid_torch))
        lossCurve_eval.append(lossEval)

    if shouldDisplayPlot:
        plotLoss(lossCurve_train, lossCurve_eval)

    hp_min = (X_train["horsepower_scaled"].min() * std_hp) + mean_hp
    hp_max = (X_train["horsepower_scaled"].max() * std_hp) + mean_hp

    disp_min = (X_train["displacement_scaled"].min() * std_disp) + mean_disp
    disp_max = (X_train["displacement_scaled"].max() * std_disp) + mean_disp

    x_surf, y_surf = np.meshgrid(
        np.linspace(hp_min, hp_max, 100), np.linspace(disp_min, disp_max, 100)
    )

    x_grid = pd.DataFrame(
        {
            "Horsepower": (x_surf.ravel() - mean_hp) / std_hp,
            "Displacement": (y_surf.ravel() - mean_disp) / std_disp,
        }
    )

    yPredicted = generatePredictions(
        model, data=torch.from_numpy(x_grid.values).view(-1, 2).float()
    ).numpy()

    if shouldDisplayPlot:
        plot_horsepowerDisplacement(x_surf, y_surf, yPredicted)

    with torch.no_grad():
        testResults = criterion(model(X_test_hp_dp), y_test_torch).numpy()
    return lossCurve_train, lossCurve_eval, testResults


def dnnModel_1D(X_train, y_train_torch, y_valid_torch, shouldDisplayPlot=False):
    model = Regressor_DNN(inFeatures=1, outFeatures=1, intermediate=32)
    getModelSummary(model, 1)

    optimiser = optim.Adam(model.parameters(), lr=0.1)

    lossCurve_train = []
    lossCurve_eval = []

    for epoch in range(500):
        lossTrain = trainModel(model, optimiser, criterion, (X_train_hp, y_train_torch))
        lossCurve_train.append(lossTrain)

        lossEval = evaluateModel(model, criterion, (X_valid_hp, y_valid_torch))
        lossCurve_eval.append(lossEval)

    if shouldDisplayPlot:
        plotLoss(lossCurve_train, lossCurve_eval)

    #  Model Prediction
    x = torch.linspace(
        X_train["horsepower"].min(),
        X_train["horsepower"].max(),
        len(X_train["horsepower"]),
    )
    yPredicted = generatePredictions(
        model, data=((x.view(-1, 1) - mean_hp) / std_hp)
    ).numpy()

    if shouldDisplayPlot:
        plot_horsepower(x, yPredicted)

    with torch.no_grad():
        testResults = criterion(model(X_test_hp), y_test_torch).numpy()
    return lossCurve_train, lossCurve_eval, testResults


def dnnModel_2D(X_train, y_train_torch, y_valid_torch, shouldDisplayPlot=False):
    model = Regressor_DNN(inFeatures=2, outFeatures=1, intermediate=32)
    getModelSummary(model, 2)

    optimiser = optim.Adam(model.parameters(), lr=0.1)

    lossCurve_train = []
    lossCurve_eval = []

    for epoch in range(500):
        lossTrain = trainModel(
            model, optimiser, criterion, (X_train_hp_dp, y_train_torch)
        )
        lossCurve_train.append(lossTrain)

        lossEval = evaluateModel(model, criterion, (X_valid_hp_dp, y_valid_torch))
        lossCurve_eval.append(lossEval)

    if shouldDisplayPlot:
        plotLoss(lossCurve_train, lossCurve_eval)

    hp_min = (X_train["horsepower_scaled"].min() * std_hp) + mean_hp
    hp_max = (X_train["horsepower_scaled"].max() * std_hp) + mean_hp

    disp_min = (X_train["displacement_scaled"].min() * std_disp) + mean_disp
    disp_max = (X_train["displacement_scaled"].max() * std_disp) + mean_disp

    x_surf, y_surf = np.meshgrid(
        np.linspace(hp_min, hp_max, 100), np.linspace(disp_min, disp_max, 100)
    )

    x_grid = pd.DataFrame(
        {
            "Horsepower": (x_surf.ravel() - mean_hp) / std_hp,
            "Displacement": (y_surf.ravel() - mean_disp) / std_disp,
        }
    )

    yPredicted = generatePredictions(
        model, data=torch.from_numpy(x_grid.values).view(-1, 2).float()
    ).numpy()

    if shouldDisplayPlot:
        plot_horsepowerDisplacement(x_surf, y_surf, yPredicted)

    with torch.no_grad():
        testResults = criterion(model(X_test_hp_dp), y_test_torch).numpy()
    return lossCurve_train, lossCurve_eval, testResults


lossTrainLinear1, lossEvalLinear1, testResultsLinear1 = linearModel_1D(
    X_train,
    y_train_torch,
    y_valid_torch,
    shouldDisplayPlot=False,
)
lossTrainLinear2, lossEvalLinear2, testResultsLinear2 = linearModel_2D(
    X_train,
    y_train_torch,
    y_valid_torch,
    shouldDisplayPlot=False,
)
lossTrainDnn1, lossEvalDnn1, testResultsDnn1 = dnnModel_1D(
    X_train,
    y_train_torch,
    y_valid_torch,
    shouldDisplayPlot=False,
)
lossTrainDnn2, lossEvalDnn2, testResultsDnn2 = dnnModel_2D(
    X_train,
    y_train_torch,
    y_valid_torch,
    shouldDisplayPlot=False,
)


# Plot all the loss curves
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(lossTrainLinear1, label="linear_1d")
plt.plot(lossTrainLinear2, label="linear_2d")
plt.plot(lossTrainDnn1, label="dnn_1d")
plt.plot(lossTrainDnn2, label="dnn_2d")
plt.ylim([0, 30])
plt.ylabel("Error [MPG]")
plt.title("Loss per Epoch")
plt.legend()
plt.grid(True)

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 2)
plt.plot(lossEvalLinear1, label="linear_1d")
plt.plot(lossEvalLinear2, label="linear_2d")
plt.plot(lossEvalDnn1, label="dnn_1d")
plt.plot(lossEvalDnn2, label="dnn_2d")
plt.ylim([0, 30])
plt.xlabel("Epoch")
plt.ylabel("Error [MPG]")
plt.title("Val loss per Epoch")
plt.legend()
plt.grid(True)

plt.show()
plt.close()


test_results = {
    "linearModel_1D": testResultsLinear1,
    "linearModel_2D": testResultsLinear2,
    "DnnModel_1D": testResultsDnn1,
    "DnnModel_2D": testResultsDnn2,
}
results = pd.DataFrame(test_results, index=["MAE"]).T
print(results)
