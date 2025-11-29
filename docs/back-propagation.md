# Back Pro-pagation
<!-- Copied from OpenCV -->
The goal here is to compute the gradient of the loss with respect to every weight in the network. This gradient indicates how much each weight contributed to the error.

Starting from the output layer, we compute the gradient of the loss with respect to the output of the network. This gradient is then propagated backward layer by layer. For fully connected layers, we use the chain rule to compute the gradient with respect to the weights and the outputs of the previous layer.
For convolutional layers, the gradient computation is a bit more involved but conceptually similar. It's like doing a convolution operation but with the gradients. Activation functions and pooling layers also have their specific gradient computations.
Once we have the gradient of the loss with respect to every weight, we can update the weights using gradient descent (or its variants like Adam, RMSProp, etc.). The weights are adjusted in the direction that reduces the loss.
In simpler terms, the neurons in the fully connected layers do not learn in isolation. Their learning is influenced by the error from the output, and this error information is sent backward through the network, adjusting weights in both convolutional and fully connected layers.

I hope this solves clarifies your concept of back-propagation in the Neural Network architecture.
