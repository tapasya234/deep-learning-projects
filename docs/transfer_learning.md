# Transfer Learning

## Feature Traning

The general approach to **fine-tuning** a convolutional base is to first load the pre-trained weights for the model and then selectively allow the last few convolutional layers in the base to be trainable. The process of selectively specifying which layers are trainable and which ones are not is often referred to as freezing or un-freezing layers via the model's **trainable** attribute.

This approach is called fine-tuning because it makes small adjustments to the more abstract representations of the model being reused to make them more relevant for the problem at hand.

Ideally, you should train the dense layers first (keeping the entire convolutional base frozen) as we did in the case of the "transfer learning with feature extraction experiment" and then start tuning the convolutional layers at a lower learning rate. However, in order to keep things simple in this example, we will train the model just once, which will include the last few layers of the convolutional base as well as the dense classifier.

In practice, when fine-tuning a model, it is better to start with fine-tuning just a few layers at a time to see how the model responds. As we will see in this example, tuning both the convolutional base and the dense layers at a lower learning rate works without any issues. However, if you are training the full convolutional base for fine-tuning, you should try to train the dense layers first and then start fine-tuning.

### How to freeze only a few layers?

There are two ways to specify which layers in the model are trainable (tunable).

We can start by making the entire convolutional base trainable by setting the trainable flag to True. Then loop over the initial layers and make them untrainable by setting the same (trainable) flag for each layer to False.

We can freeze the entire convolutional base by setting the trainable flag to False, and then loop over the last few layers and set the trainable flag to True.
