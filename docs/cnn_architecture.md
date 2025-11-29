# CNN Architecture

<!-- Copied from Reddit, need to gather more info and update it -->
Any advice for how to go about designing CNN architecture?
Discussion
When should you add another convolutional layer?

When should you add another pooling layer?

How should you choose the kernel, stride size, padding, or dilation of convolutional layers? Or the size of averaging windows?

When should you use max vs average pooling?

When should you use global pooling?

Is it ever useful to put dense layers in the middle of the network, instead of only as the final layer?

With neural nets and especially CNNs there are way more possible configurations than with other ML models and I don't know how to choose or optimize them. I've learned the math and theory behind how CNNs work and I thought that would help, but now that I'm trying real world problems I find that I still don't know how to make effective networks. Can anyone can give any tips or resources you find helpful?

Hey OP,

It's a great question. If you're curious about diving deeper, I found this survey paper that covers a lot of the specifics of new-ish architectures

<https://arxiv.org/abs/1901.06032>

Before I get into it, common practice for years has been to use transfer learning from off-the-shelf networks to whatever your specific use-case is. That said, I can give my two cents on the topic.

Let's limit ourselves to vision networks, as they are probably the easiest to think about and you mentioned convolutions (though, convolutions appear in many non-vision architectures as well). We have four basic building blocks: convolutions, pooling, fully-connected (fc) and activation layers.

The general flow, following mostly from the LeNet architecture and derivatives (like alexnet, both of which are found in the paper above) tends to be:

Conv -> pool -> activation -> (repeat n) -> flatten -> fc

The idea behind this is:

convolutions extract features from their input maps

pooling both (1) reduces dimensionality and (2) filters information, favoring higher activations of features and discarding unimportant information

activation adds nonlinearity, this enables the network to discover hidden, high dimensional representations of inputs, and also helps with information filtering (in the case of ReLU, for example, which discards anything negative and only keeps positive activations)

The fully connected layer brings all of these layers together and leverages these high dimensional, but information-dense features, and performs a classification in much the same way as an SVM would, by learning a projection such that the input features for a particular class maximize the output neuron representing it (technically this is dependent on the loss function and other things but bear with me).

As for the intuition on what layers to place where, how deep they should be, etc. Unfortunately there isn't exactly a clear cut answer, but given the above and other math facts, we can glean a few things for what not to do.

2+ convolution layers in a row is equivalent to one big convolution because both are linear operations

Pooling or convolution strides should be used to reduce dimensionality, but take care not to discard too much information too quickly

Convolutions learn features, and you can learn as many feature maps as you want from a given layer, but if you don't also downsample every few layers, you won't be able to learn complex features because your convolutions will only be taking in information from small parts of the image as a whole.

Double descent (worth a Google) and empirical evidence suggests that it's probably okay to have an over parameterized network (too big), but there is an inflection point where you end up either unable to learn or you overfit to your dataset. Take care not to end up here and experiment with a wide variety of architecture sizes.

There are a ton of other subjects on neural network design. I'll list some of them here and you can Google at your leisure

residual layers/skip connections (help with gradient descent for deep networks)

attention layers

conditional networks (gating)

autoencoder architecture design

strided, grouped, and dilated convolutions

batch norm/layer norm

global average pooling

activations (softmax, sigmoid, ReLU)

auxiliary outputs (and associated loss functions)

Hope that helps!
