# The aim of this assignment is to understand the Gradient descent optimization process in more detail.
# In the previous section on Gradient Descent, we discussed how to implement gradient calculation and weight update
# for a single variable ( m ) using simple math operations.
# In this assignment are required to do the same, but for 2 variables.
# We will use the full form of a line i.e. y = mx + c. You need to estimate the values of the two variables
# m and c using Stochastic Gradient Descent.

# You have two tasks:
# Implement the gradient calculation step for the 2 variables.
# Implement the weight update step for the 2 variables.

import torch
import matplotlib.pyplot as plt

plt.style.use("ggplot")
torch.manual_seed(0)

# Generate Data
# We will generate 1000 data points for the experiment. The x axis is the independent variable which has values
# randomly distributed between -5 to 5. We assume some values for m and c to create the data points for the
# dependent variable ( y-axis ). We also add some randomness so that the y values are different for the same x.


# Now, we have a simple dataset which has been generated using a linear model in the presence of noise.
# The data has been dispayed using the scatter plot.
# Generating y = mx + c + random noise
def generateData():
    num_data = 1000

    # True values of m and c
    m_line = 3.3
    c_line = 5.3

    # input (Generate random data between [-5,5])
    x = 10 * torch.rand(num_data) - 5

    # Output (Generate data assuming y = mx + c + noise)
    y_label = m_line * x + c_line + torch.randn_like(x)
    y = m_line * x + c_line

    # Plot the generated data points
    plt.plot(x, y_label, ".", color="g", label="Data points")
    plt.plot(x, y, color="b", label="y = mx + c", linewidth=3)
    plt.ylabel("y")
    plt.xlabel("x")
    plt.legend()
    plt.show()

    return x, y


x, y = generateData()

# We calculate the loss function and then take partial derivatives w.r.t m and c respectively.
# `l = ∑(𝑦𝑖−𝑚𝑥𝑖−𝑐)^2`

# `∂𝑙/∂𝑚 = -2 * ∑𝑥𝑖(𝑦𝑖−𝑚𝑥𝑖−𝑐)`

# `∂𝑙/∂𝑐 = -2 * ∑(𝑦𝑖−𝑚𝑥𝑖−𝑐)`

# To follow the slope of the curve, we need to move m in the direction of negative gradient.
# However, we need to control the rate at which we go down the slope so that we do not overshoot the minimum.
# So we use a parameter `𝜆` called the learning rate.

# m(k) =m(k-1) - 𝜆 * (∂𝑙/∂𝑚)

# c(k) = c(k-1) - 𝜆 * (∂𝑙/∂𝑐)


def gradient_wrt_m_and_c(inputs, labels, m, c, k):
    """
    All arguments are defined in the training section of this notebook.
    This function will be called from the training section.
    So before completing this function go through the whole notebook.

    inputs (torch.tensor): input (X)
    labels (torch.tensor): label (Y)
    m (float): slope of the line
    c (float): vertical intercept of line
    k (torch.tensor, dtype=int): random index of data points
    """
    # gradient w.r.t to m is g_m
    # gradient w.r.t to c is g_c
    num_samples = len(inputs)
    ###
    ### YOUR CODE HERE
    ###

    e = labels[k] - m * inputs[k] - c
    g_m = -2 * torch.sum(x[k] * e)
    g_c = -2 * torch.sum(e)

    return g_m, g_c


X = torch.tensor([-0.0374, 2.6822, -4.1152])
Y = torch.tensor([5.1765, 14.1513, -8.2802])
m = 2
c = 3
k = torch.tensor([0, 2])

gm, gc = gradient_wrt_m_and_c(X, Y, m, c, k)

print("Gradient of m : {0:.2f}".format(gm))
print("Gradient of c : {0:.2f}".format(gc))


def update_m_and_c(m, c, g_m, g_c, lr):
    """
    All arguments are defined in the training section of this notebook.
    This function will be called from the training section.
    So before completing this function go through the whole notebook.

    g_m = gradient w.r.t to m
    c_m = gradient w.r.t to c
    """
    # update m and c parameters
    # store updated value of m is updated_m variable
    # store updated value of c is updated_c variable
    ###
    ### YOUR CODE HERE
    ###
    updated_m = m - lr * g_m
    updated_c = c - lr * g_c
    return updated_m, updated_c


m = 2
c = 3
g_m = -24.93
g_c = 1.60
lr = 0.001
m, c = update_m_and_c(m, c, g_m, g_c, lr)

print("Updated m: {0:.2f}".format(m))
print("Updated c: {0:.2f}".format(c))
