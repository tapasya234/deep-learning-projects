import re
import torch

# # With `required_grad` set to true, all the operations performed on it are recorded.
# x = torch.tensor(4.0, requires_grad=True)
# y = x**2

# # Next, the 'autograd' module is used to compute the backward pass.
# # The following call will compute the gradient of loss with respect
# # to all Tensors with requires_grad=True.
# # After this call "x.grad" will hold the gradient dy/dx.
# y.backward()

# print(f"dx_dy: {x.grad}")

# # Two tensors with autograd enabled
# w1 = torch.tensor(5.0, requires_grad=True)
# w2 = torch.tensor(3.0, requires_grad=True)

# z = 3 * w1**2 + 2 * w1 * w2
# z.backward()
# print(f"dz_dw1: {w1.grad}")
# print(f"dz_dw2: {w2.grad}")


# # IMPLICIT AUTOGRAD ON SCALAR OUTPUTS
# a = torch.rand((3, 5), requires_grad=True)
# result = a * 5.0
# print(result)

# # Grad can be implicitly created only for scalar outputs.
# # So, calculate the sum here so that the output becomes a scalar and apply a backward pass.
# sum = result.sum()
# sum.backward()
# print(f"dsum_da: {a.grad}")

# DISABLED AUTOGRAD ON TENSORS
a = torch.randn((3, 5), requires_grad=True)

aDetached = a.detach()
resultDetached = aDetached * 10
result = a * 5

mean = result.sum()
mean.backward()
print(f"a.grad: \n{a.grad}")

try:
    meanDetached = resultDetached.sum()
    mean.backward()
    print(f"aDetached.grad: {aDetached.grad}")
except RuntimeError as e:
    print("\n", e)
