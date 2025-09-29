import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
block_plot = True


def createNonLinearData(
    xMin=-10,
    xMax=10,
    numData=100,
    theta0=0,
    theta1=0.3,
    theta2=0.05,
    noise=0.1,
):
    X = np.linspace(xMin, xMax, numData)
    y = (
        theta0
        + theta1 * X * np.cos(X)
        + theta2 * X**2
        + noise
        + np.random.normal(size=numData)
    )

    X = torch.from_numpy(X)
    X = X.view((len(X), 1))
    y = torch.from_numpy(y)
    y = y.view((len(y), 1))

    # Create two features from the input data that match the functional form of the data we generated above.
    Xf = torch.cat((X * torch.cos(X), X * X), axis=1).float()
    return X, y, Xf


def plotData(
    X,
    y,
    xLim=(-10, 10),
    yLim=(-10, 10),
    theta0=0,
    theta1=0,
    theta2=0,
    yPredicted=None,
):
    plt.figure()
    plt.plot(X, y, "b.")
    plt.xlabel("X")
    plt.xlim(xLim)

    plt.ylabel("Y")
    plt.ylim(yLim)

    if yPredicted is not None:
        plt.plot(X, yPredicted, "c-")

    plt.text(-5, 7.0, "theta0=" + str(theta0), fontsize=14)
    plt.text(-5, 6.0, "theta1=" + str(theta1), fontsize=14)
    plt.text(-5, 5.0, "theta2=" + str(theta2), fontsize=14)


def fitLine_LinearNN(X, y, Xf):
    model = nn.Linear(in_features=2, out_features=1)
    optimiser = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    lossValues = []
    epochs = []
    for epoch in range(10000):
        optimiser.zero_grad()
        output = model(Xf)
        loss = criterion(output, y.float())
        loss.backward()
        optimiser.step()
        lossValues.append(loss.detach().numpy().item())
        epochs.append(epoch + 1)

    plt.figure(figsize=[15, 5])
    plt.plot(epochs, lossValues, color="green", label="Training Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.xlabel("Loss")
    plt.legend()

    theta0_predicted = model.state_dict()["bias"][0].numpy()
    weights = model.state_dict()["weight"][0].numpy()
    theta1_predicted = weights[0]
    theta2_predicted = weights[1]

    printCoefficients(
        "Predicted Coofficients:",
        theta0_predicted,
        theta1_predicted,
        theta2_predicted,
    )

    plotData(
        X,
        y,
        xLim=(-12, 12),
        yLim=(-600, 10),
        theta0=theta0_predicted,
        theta1=theta1_predicted,
        theta2=theta2_predicted,
        yPredicted=model(Xf).detach().numpy(),
    )


def printCoefficients(title, theta0, theta1, theta2):
    print(title)
    print("Theta0: ", theta0)
    print("Theta1: ", theta1)
    print("Theta2: ", theta2)


theta0 = -5
theta1 = 0.15
theta2 = -5.5
X, y, Xf = createNonLinearData(theta0=theta0, theta1=theta1, theta2=theta2, noise=0.3)
plotData(
    X,
    y,
    xLim=(-12, 12),
    yLim=(-600, 10),
    theta0=theta0,
    theta1=theta1,
    theta2=theta2,
)
printCoefficients("Actual Co-efficients:", theta0, theta1, theta2)
fitLine_LinearNN(X, y, Xf)

plt.show()
