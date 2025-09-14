import numpy as np
import torch


# ReLU stands for Rectified Linear Unit. It defined as follows: z = max(0, x)
# So if the input is:
# tensor([[ 1.0000,  2.0000, -3.0000],
#         [ 2.5000, -0.2000,  6.0000]])
# You have to return the following output:
# tensor([[1.0000, 2.0000, 0.0000],
#         [2.5000, 0.0000, 6.0000]])
def ReLU(tensor):
    results = torch.zeros_like(tensor)
    print(results.shape)
    results[tensor > 0] = tensor[tensor > 0]
    return results


# test your result
a = torch.tensor([[1, 2, -3], [2.5, -0.2, 6]])
# print(ReLU(a))


# Softmax is defined as follows: `softmax(𝑧𝑖)=exp(𝑧𝑖)/∑𝑗exp(𝑧𝑗)`
# So if the input is:
# tensor([0.6000, 5.2000, 9.2000])
# Then, the function should return the following as output:
# tensor([1.8076e-04, 1.7983e-02, 9.8184e-01])
def softmax(array):
    exponentials = torch.exp(array)
    return exponentials / torch.sum(exponentials)


# Test your result
a = torch.tensor([0.6, 5.2, 9.2])
# print(softmax(a))


# Neural Network Neuron Implementation
# You have to implement the following function: `𝑌=𝑊𝑋+𝐵`
# The function will take weight `𝑊, bias `𝐵, and input`𝑋` as arguments. You have to return outputs `𝑌`.
# So, if the input is:
# W = tensor([[1.2000, 0.3000, 0.1000],
#             [0.0100, 2.1000, 0.7000]])
# B = tensor([2.1000, 0.8900])
# X = tensor([0.3000, 6.8000, 0.5900])
# Then, the function should return:
# tensor([ 4.5590, 15.5860])
def neural_network_neurons(W, B, X):
    return torch.matmul(W, X) + B
    # for i in range(W.shape[0]):
    #     print(W[i] * X)
    #     print(torch.sum(W[i] * X))
    #     output[i] = torch.sum(torch.dot(W[i], X)) + B[i]
    # return output


# test your code
W = torch.tensor([[1.2, 0.3, 0.1], [0.01, 2.1, 0.7]])
B = torch.tensor([2.1, 0.89])
X = torch.tensor([0.3, 6.8, 0.59])

print(neural_network_neurons(W, B, X))
