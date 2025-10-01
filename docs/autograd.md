# Automatic Differentiation with Autograd

`torch.autograd` is a submodule in PyTorch that provides automatic differentiation capabilities. It is a key component for training deep learning models, as it simplifies the process of computing gradients of tensors with respect to model parameters. These gradients are then used to update the model parameters during the optimization process (e.g., using stochastic gradient descent).

The main uses of torch.autograd include:

- Automatic differentiation: It simplifies the process of computing gradients for complex models with many layers and parameters. By keeping track of the computation graph during forward propagation, torch.autograd can compute the gradients efficiently using the chain rule during backward propagation.

- Gradient tracking: torch.autograd enables gradient tracking by wrapping tensors with the torch.Parameter class, which has a built-in attribute called requires_grad. When this attribute is set to True, the tensor starts tracking all the operations performed on it, and the gradients can be calculated during the backward pass.

- Backward propagation: The backward pass can be initiated by calling the backward() function on a scalar tensor (usually the loss). This function computes the gradients for all tensors with requires_grad=True in the computation graph.

- Gradient accumulation: When the gradients are calculated, they are stored in the grad attribute of the tensors involved in the computation. These gradients can then be accessed and used for updating the model parameters.

- Dynamic computation graph: torch.autograd supports dynamic computation graphs, which means that the graph is constructed on-the-fly during a forward pass and can be modified at runtime. This allows for greater flexibility in model design and makes it easier to implement certain types of models, such as recurrent neural networks (RNNs) and models with control flow.

In summary, torch.autograd is a core component of PyTorch that facilitates automatic differentiation, enabling the efficient training of deep learning models. It tracks tensor operations, computes gradients during the backward pass, and accumulates these gradients, which can be used to update the model parameters during optimization.

## Disabling Autograd for tensors

Suppose there's a scenario where we don't need to compute gradients for all the variables involved in the pipeline. In that case, the PyTorch API provides 2 ways to disable autograd.

- `detach` - returns a copy of the tensor with autograd disabled. This copy is built on the same memory as the original tensor, so in-place size / stride / storage changes (such as resize_/ resize_as_/ set_ / transpose_) modifications are not allowed.
- `torch.no_grad()` - It is a context manager that allows you to guard a series of operations from autograd without creating new tensors.
