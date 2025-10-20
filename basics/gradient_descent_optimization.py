import torch
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# Gradient descent is a gradient-based optimization algorithm that is used extensively in machine learning
# and deep learning to minimize a loss function by iteratively adjusting the model parameters in the
# direction of steepest descent based on the negative gradient.
# Specifically, we will look at how to fit a straight line through a set of points to determine
# the slope of the line. To do this, we will define a loss function that quantifies the error between
# the data and the mathematical model we choose to represent the data,
# and we will use this loss function to develop an update rule that will iteratively converge to the optimal value.
# We will conclude the notebook with a variation on the Gradient Descent algorithm called
# Mini-Batch Stochastic Gradient Descent which is the basis for training neural networks.

# Constant for plotting loss
MAX_LOSS = 30.0

# Constants used for finding the best `m` value
LEARNING_RATE = 0.005
NUM_ITERATIONS = 50
M_DEFAULT_VALUE = 2


def plotLossModel(loss, xLim, yLim, title):
    plt.figure()
    plt.plot(loss.numpy(), "c--")
    plt.xlim(xLim)
    plt.ylim(yLim)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(title)


def plotLinearModel(x, y, m, title, xlim=(0, 10), ylim=(0, 10)):
    xMin = torch.min(x)
    xMax = torch.max(x)
    yMin = torch.min(y)
    yMax = torch.max(y)

    xPlot = np.linspace(xMin.item(), xMax.item(), 2)
    yPlot = m * xPlot

    plt.figure()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.plot(xPlot, yPlot, "c--")
    plt.scatter(x, y, color="blue", s=20)
    plt.xlabel("X")
    plt.ylabel("Y")

    if title is not None:
        plt.title(title)

    xc = 0.05 * (xMax - xMin)
    yc = 0.95 * (yMax - yMin)
    plt.text(xc, yc, "Slope: " + str(int(m * 1000) / 1000), fontsize=14)


def createData(numDataPoints=30):
    torch.manual_seed(42)

    # Create some data that is roughly linear (but not exactly).
    x = 10 * torch.rand(numDataPoints)
    y = x + torch.randn(numDataPoints) * 0.3

    return x, y


x, y = createData()

# Plot the sample data and the initial guess for the line by generating the data for slope of 2
plotLinearModel(x, y, M_DEFAULT_VALUE, title="Sample Data with Initial Line")


def bruteForceSearch():
    minValue = 0.0
    maxValue = 2.0

    # Number of steps between `minValue` and `maxValue`
    numSteps = 20
    stepSize = (maxValue - minValue) / (numSteps - 1)

    # Space to store all values of `m` and the `loss` corresponding to each `m`
    m = torch.zeros(numSteps)
    loss = torch.zeros(numSteps)

    for i in range(numSteps):
        m[i] = minValue + i * stepSize
        e = y - m[i] * x
        loss[i] = torch.sum(torch.matmul(e, e)) / len(x)

    minIndex = torch.argmin(loss)

    print("Brute Force Search")
    print(f"Minimum Loss: {loss[minIndex]}")
    print(f"Best value of m: {m[minIndex]}")

    # Plot `m` vs `loss`
    plt.figure()
    plt.plot(m.numpy(), loss.numpy(), "c--")
    plt.xlabel("M")
    plt.ylabel("Loss")
    plt.title("Brute Force Search")

    return m[minIndex]


# The 'loss' plot from the previous section shows that we can start from an initial guess of `m`,
# and follow the slope of the curve downward to reach the bottom of the curve. To automate this process,
# for a given value of `m` we can compute the gradient of the loss function and use that value to inform us how to adjust `m`.
# If the gradient is positive, then we'll need to lower the value of `m` to move closer to the minimum, and '
# 'if the gradient is negative, we'll need to raise the value of `m`. This simple idea is called Gradient Descent.
def gradientDescent(numIterations, learningRate, m):
    loss = torch.zeros(numIterations)
    slope = torch.zeros(numIterations)

    xLen = len(x)
    for i in range(numIterations):
        e = y - m * x
        loss[i] = torch.sum(torch.matmul(e, e)) / xLen
        gradient = -2.0 * torch.sum(x * e) / xLen
        m = m - learningRate * gradient
        slope[i] = m

    lossMin, indexMin = torch.min(loss, 0)

    print("Gradient Descent")
    print("Best iteration: ", indexMin.numpy())
    print("Minimum loss:   ", lossMin.numpy())
    print("Best parameter: ", slope[indexMin].numpy())

    plotLossModel(
        loss,
        xLim=(0, numIterations),
        yLim=(0, numIterations),
        title="Gradient Descent",
    )

    return slope[indexMin]


# In the real world, calculating the gradient based on all data points can be computationally expensive.
# Fortunately, using all the data points for computing the gradient is unnecessary.
# Stochastic Gradient Descent is an updated algorithm which uses a single randonly choosen data point to compute the gradient at each iteration.
# Even though the gradient at each step is not as accurate, the idea still works.
# The convergence might be slower using this technique because the gradient is not as accurate.
def stochasticGradientDescent(numIterations, learnignRate, m):
    loss = torch.zeros(numIterations)

    for i in range(numIterations):
        e = y - m * x
        loss[i] = torch.sum(torch.matmul(e, e)) / len(x)

        # Randomly select a training data point
        k = torch.randint(0, len(y), (1,))[0]
        gradient = -2.0 * x[k] * (y[k] - m * x[k])

        m = m - learnignRate * gradient

    print("Stochastic Gradient Descent")
    print("Final loss:   ", loss[-1].numpy())
    print("Final parameter: ", m.numpy())

    plotLossModel(
        loss,
        xLim=(0, numIterations),
        yLim=(0, MAX_LOSS),
        title="Stochastic Gradient Descent",
    )
    return m


# Using more than one data point for the gradient calculation has two advantages:
#  - Using multiple data points produces a more accurate estimate for the gradient.
#  - GPUs are highly efficient at processing gradient computations.
# So, we get better results and faster convergence if we use a small batch of data points, called a mini-batch, to compute the gradients.
# A "mini-batch" approach strikes a nice balance between using all the data points vs just a single data point.
def stochasticGD_miniBatch(numIterations, learningRate, m, batchSize):
    loss = torch.zeros(numIterations)

    for i in range(numIterations):
        # Randomly select a batch of data points
        k = torch.randint(0, len(y), (batchSize,))

        e = y[k] - m * x[k]
        loss[i] = torch.sum(torch.matmul(e, e)) / batchSize

        gradient = (-2.0 / batchSize) * torch.sum(x[k] * (y[k] - m * x[k]))
        m = m - learningRate * gradient

    print("Stochastic Gradient Descent, Mini Batch")
    print("Final loss:   ", loss[-1].numpy())
    print("Final parameter: ", m.numpy())

    plotLossModel(
        loss,
        xLim=(0, NUM_ITERATIONS),
        yLim=(0, MAX_LOSS),
        title="Stochastic Gradient Descent, Mini Batch",
    )
    return m


def gradientDescent_autograd(numIterations, learningRate, m: torch.nn.Parameter):
    losses = torch.zeros(numIterations)
    slopes = torch.zeros(numIterations)

    for i in range(numIterations):
        loss = ((y - m * x) ** 2).mean()

        # Automatically compute the gradient of the loss with respect to parameter "m".
        loss.backward()

        dl_dm = m.grad

        # Manually update weights using gradient descent, but make sure this change isn't tracked by autograd.

        # --- Method 1 ---
        # Modiy the underlying storage by performing inplace operation on the .data property.
        # m.data -= learningRate * dl_dm
        # m.grad.zero_()

        with torch.no_grad():
            # --- Method 2 ---
            # Create a new tensor and copy the elements from the new tensor into 'm'.
            # This method can be used in case inplace operations cannot be performed.
            # mNew = m - learningRate * dl_dm
            # m.copy_(mNew)

            # --- Method 3 ---
            # Perform the updates to the tensor and manually zero or set ".grad" to None.
            # m.grad.zero_()
            m -= learningRate * dl_dm
            m.grad = None

        losses[i] = loss.detach()
        slopes[i] = m.detach()

    lossMin, indexMin = torch.min(losses, 0)

    print("Gradient Descent, Auto Grad")
    print("Final loss:   ", lossMin.numpy())
    print("Final parameter: ", slopes[indexMin].numpy())

    plotLossModel(
        losses,
        xLim=(0, NUM_ITERATIONS),
        yLim=(0, MAX_LOSS),
        title="Gradient Descent, Auto Grad",
    )
    return slopes[indexMin]


mBruteForce = bruteForceSearch()
plotLinearModel(x, y, mBruteForce, "Brute Force Search")

mGradientDescent = gradientDescent(NUM_ITERATIONS, LEARNING_RATE, M_DEFAULT_VALUE)
plotLinearModel(x, y, mGradientDescent, "Gradient Descent")

mStochasticGD = stochasticGradientDescent(
    NUM_ITERATIONS, LEARNING_RATE, M_DEFAULT_VALUE
)
plotLinearModel(x, y, mStochasticGD, "Stochastic Gradient Descent")

mSGD_MB = stochasticGD_miniBatch(NUM_ITERATIONS, LEARNING_RATE, M_DEFAULT_VALUE, 10)
plotLinearModel(x, y, mSGD_MB, "Stochastic Gradient Descent, Mini Batch")

mAutoGragInitial = torch.nn.Parameter(
    torch.tensor(M_DEFAULT_VALUE, dtype=torch.float32)
)
mGD_autoGrad = gradientDescent_autograd(NUM_ITERATIONS, LEARNING_RATE, M_DEFAULT_VALUE)
plotLinearModel(x, y, mGD_autoGrad, "Gradient Descent, Auto Grad")

plt.show()
