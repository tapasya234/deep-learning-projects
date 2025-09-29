import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

# Regression is a form of supervised learning which aims to model the relationship
# between one or more input variables (features) and a continuous (target) variable.
# It can be assumed that for a linear regression, the relationship between the
# input variables(x) and target variable(y) can be expressed as a
# weighted sum of the inputs. In short, linear regression aims to learn a function
# that maps one or more input features to a single numerical target value.

# Since linear regression can be modeled as a linear neural network,
# it provides an excellent running example to introduce the essential
# components of neural networks.

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
block_plot = False

# Random manual seed for consistency.
seed = 54
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.manual_seed(seed)
torch.backends.deterministic = True
torch.backends.benchmark = True


# Create some linear data with a small amount of noise
def createLinearData(numData=100, yOffset=0, slope=1, stdDev=0.3):
    X = 10 * torch.rand(size=[numData])
    y = yOffset + slope * X + torch.normal(std=stdDev, mean=0, size=[numData])

    X = X.view((len(X), 1))
    y = y.view((len(y), 1))

    return X, y


def plotData(X, y, yIntercept, slope, xLim=(0, 10), yLim=(0, 10), yPredicted=None):
    plt.figure()
    plt.plot(X, y, "b.")
    if yPredicted is not None:
        plt.plot(X, yPredicted, "c-")
    plt.xlabel("X")
    plt.xlim(xLim)
    plt.ylabel("Y")
    plt.ylim(yLim)
    # plt.show(block=block_plot)
    plt.text(1, 7.0, "Y-Intercept: " + str(yIntercept), fontsize=14)
    plt.text(1, 6.5, "Slope: " + str(slope), fontsize=14)


def printPredictedValues(slopePredicted, yPredicted):
    print("Predicted Co-efficients: ")
    print("y-intercept: ", yPredicted)
    print("Slope: ", slopePredicted)


def computeTheta(X, y):
    m = X.shape[0]  # NUmber of samples

    # Concat 1 to the beginning of every feature vector to represent t
    # he value to be multipled by theta(0).
    X = torch.cat((torch.ones(m, 1), X), axis=1)
    y = y.view((m, 1))

    XTranspose = X.T
    XT_X = torch.matmul(XTranspose, X)
    XT_X_inverse = torch.inverse(XT_X)
    XT_y = torch.matmul(XTranspose, y)

    return torch.matmul(XT_X_inverse, XT_y)


def predictY(X, theta):
    X = torch.cat((torch.ones(X.shape[0], 1), X), axis=1)
    return torch.matmul(X, theta)


def fitLine_NormalEquations(X, y):
    theta = computeTheta(X, y)
    yIntercept = theta[0][0].numpy()
    slope = theta[1][0].numpy()

    printPredictedValues(slopePredicted=slope, yPredicted=yIntercept)
    # plotData(
    #     X,
    #     y,
    #     yIntercept,
    #     slope,
    #     yPredicted=predictY(X, theta),
    # )

    # As evident, using the normal equations to solve linear regression problems
    # is very simple. However, solving this equation requires inverting a matrix
    # which can be computationally expensive, especially for very large problems
    # which may include thousands of features.
    # There is also an issue associated with its stability, it is possible that
    # the matrix is not invertible due to numerical issues.


def fitLine_LinearNN(X, y, shouldIncludeBias=False):
    # The following steps summarize the workflow in PyTorch:
    #  - Build/Define a network model using PyTorch.
    #  - Define the optimizer.
    #  - Define the loss to be used.
    #  - Train the model.
    #  - Predict the output model(Test Data).
    model = nn.Linear(in_features=1, out_features=1, bias=shouldIncludeBias)
    optimiser = optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()

    lossCurve = []
    epochs = []
    for epoch in range(10000):
        optimiser.zero_grad()
        output = model(X.float())
        loss = criterion(output, y.float())
        loss.backward()
        optimiser.step()
        lossCurve.append(loss.detach().numpy().item())
        epochs.append(epoch + 1)

    plt.figure(figsize=[15, 5])
    plt.plot(epochs, lossCurve, color="orange", label="Training Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.xlabel("Loss")
    plt.legend()

    slope = model.state_dict()["weight"][0]
    yIntercept = None
    if shouldIncludeBias:
        yIntercept = model.state_dict()["bias"][0]
    yPredicted = model(X.float()).detach().numpy()

    printPredictedValues(slopePredicted=slope, yPredicted=yIntercept)
    # plotData(
    #     X,
    #     y,
    #     yIntercept,
    #     slope,
    #     yPredicted=yPredicted,
    # )


yIntercept = 3.5
slope = 0.4
X, y = createLinearData(yOffset=yIntercept, slope=slope, stdDev=0.3)
plotData(X, y, yIntercept, slope)

print("Actual Co-efficients: ")
print("y-intercept: ", yIntercept)
print("Slope: ", slope)

fitLine_NormalEquations(X, y)
fitLine_LinearNN(X, y, shouldIncludeBias=False)
fitLine_LinearNN(X, y, shouldIncludeBias=True)
plt.show()
