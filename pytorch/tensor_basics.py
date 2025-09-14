import torch
import numpy as np

# TENSOR CREATION

# The basic data type for all Deep Learning-related operations is float
zeros = torch.zeros(3, 4)
print("Zeros: ")
print(zeros)

random = torch.rand(2, 2, 3, 4)
print("Random: ")
print(random)

# PYTHON/NUMPY/PYTORCH INTEROPERABILITY

# Simple Python list
pythonList = [1, 2]
# Create a numpy array from python list
numpyArray = np.array(pythonList)

# Create a torch Tensor from python list
tensorFromList = torch.tensor(pythonList)
# Create a torch Tensor from Numpy array
tensorFromArray = torch.tensor(numpyArray)
# Another way to create a torch Tensor from Numpy array (Share same storage)
tensorFromArrayV2 = torch.from_numpy(numpyArray)

# Convert torch tensor to numpy array
arrayFromTensor = tensorFromArray.numpy()
arrayFromTensorV2 = tensorFromArrayV2.numpy()

print("List:   ", pythonList)
print("Array:  ", numpyArray)
print("TensorFromList: ", tensorFromList)
print("TensorFromArray: ", tensorFromArray)
print("TensorFromArrayV2: ", tensorFromArrayV2)
print("ArrayFromTensor:  ", arrayFromTensor)
print("ArrayFromTensorV2:  ", arrayFromTensor)

# Differences between `tensorFromArray` and `tensorFromArrayV2`
# `tensor` copies memory whereas `from_numpy` shares the same underlying storage
print("----------")
numpyArray[0] = 123
print("Updated Array: ", numpyArray)
print("Updated TensorFromArray: ", tensorFromArray)
print("Updated TensorFromArrayV2: ", tensorFromArrayV2)

print("----------")
tensorFromArray[1] = 666
print("Updated Array: ", numpyArray)
print("Updated TensorFromArray: ", tensorFromArray)
print("Updated TensorFromArrayV2: ", tensorFromArrayV2)

print("----------")
tensorFromArrayV2[1] = 999
print("Updated Array: ", numpyArray)
print("Updated TensorFromArray: ", tensorFromArray)
print("Updated TensorFromArrayV2: ", tensorFromArrayV2)

# TENSOR SHAPES

# We can change the shape of a tensor without the memory copying overhead. There are two methods for that: `reshape` and `view`.
# The difference is the following:
#  - `view` tries to return the tensor, and it shares the same memory with the original tensor.
#    In case, if it cannot reuse the same memory due to some reasons, it just fails.
#  - `reshape` always returns the tensor with the desired shape and tries to reuse the memory. If it cannot, it creates a copy.

tensor = torch.rand(2, 3, 4)
print("Original: ", tensor)
print("Pointer to data", tensor.data_ptr())
print("Shape", tensor.shape)
print("----------")
reshaped = tensor.reshape(24)
print("Reshaped: ", reshaped)
print("Pointer to data", reshaped.data_ptr())
print("Shape", reshaped.shape)
print("----------")
reshapedV2 = tensor.reshape(2, 4, 3)
print("ReshapedV2: ", reshapedV2)
print("Pointer to data", reshapedV2.data_ptr())
print("Shape", reshapedV2.shape)
print("----------")
view = tensor.view(3, 2, 4)
print("View: ", view)
print("Pointer to data", view.data_ptr())
print("Shape", view.shape)

assert tensor.data_ptr() == view.data_ptr()
assert np.all(np.equal(tensor.numpy().flat, reshaped.numpy().flat))
assert np.all(np.equal(tensor.numpy().flat, reshapedV2.numpy().flat))

# Multidimensional tensors are stored and accessed in a contiguous 1D memory block.
# Strides determine how many memory positions you need to skip to move to the next element along each dimension
# When a tensor is transposed, the data is not rearranged in memory, but the strides change to reflect the new access pattern.
print("Original stride", tensor.stride())
print("Reshaped stride", reshaped.stride())
print("ReshapedV2 stride", reshapedV2.stride())
print("View stride", view.stride())


# Many operations on tensors have an in-place form that does not return modified data but changes values in the tensor.
# The in-place version of the operation has a trailing underscore according to PyTorch's naming convention - in the example above, it is clamp_.

# IMAGES IN PYTORCH

# In PyTorch, the convention for the order of dimensions in tensors differs from some other frameworks.
# Specifically, the channel dimension is typically placed first. This is known as the "channels first" format.
# For example, an image tensor in PyTorch is usually structured as (N, C, H, W), where:
# N is the batch size, representing the number of images.
# C is the number of channels, such as 3 for RGB images.
# H is the height of the image.
# W is the width of the image.
