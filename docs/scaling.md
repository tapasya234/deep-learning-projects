# Scaling

One of the steps in data pre-processing is scaling and it is definitely recomended when the dataset contains a wide range of feature data values. It is often recommended to scale features so that they span a similar value range since the features are multipled by model weights so, the scale of the output and gradients are affected by the scale of the inputs.

Although a model might converge without feature scaling, scaling features makes training much more stable and also facilitates the optimization process by allowing gradient descent to converge much faster.

Normalisation and Standardisation are two common techniques used to scale data, but the choice between them depends on the nature of user's data and the requirements of the model.

## Normalisation

Normalisation typically refers to scaling the data to a fixed range, usually [0, 1] or [-1, 1]. This is often done using the min-max scaling method, where each feature is scale to the range [0,1].

```
                x(i) - min(x(i))
    x(i) =  ------------------------
              max(x(i)) - min(x(i))
```

### When to Use Normalisation

- Uniform Distribution: If the data does not follow a Gaussian distribution and has a skewed distribution, normalisation can be more appropriate.

- Machine Learning Algorithms: Algorithms like k-nearest neighbors (KNN) and neural networks, especially when using activation functions like sigmoid or tanh, can benefit from normalised data.

- Bounding Values: When there is a need to bound the values within a specific range, such as when dealing with pixel values in image processing.

When training a CNN for image classification, normalizing the pixel values can help in faster convergence and better performance. For instance, when working with the CIFAR-10 dataset, which contains images with pixel values ranging from 0 to 255, a user might normalise these values to the range [0, 1].

## Standardisation (Z-score Scaling)

Standardising  assumes the original data is normally distributed and scales the feature to have zero mean and a standard deviation of 1. This is accomplished for each feature (x(i))  by subtracting the mean of the feature data from each data point (referred to as mean subtraction) and then dividing that result by the standard deviation for the feature data as shown below:

```
                x(i) - mean(x(i))
    x(i) =  ------------------------
                  std-dev(x(i))
```

### When to Use Standardisation

- Normal Distribution: If your data is normally distributed, standardisation can be more appropriate.

- Machine Learning Algorithms: Algorithms like linear regression, logistic regression, and support vector machines (SVM) often perform better with standardised data because they assume the input features are normally distributed and have similar scales.

- Distance-Based Algorithms: Algorithms that rely on distance metrics, such as K-means clustering and principal component analysis (PCA), typically perform better with standardised data.

For tasks like image segmentation, where pixel-level classification is performed, Standardisation can help in achieving better results by ensuring that all features contribute equally to the learning process

## Choosing Between Normalisation and Standardisation

- Data Distribution: If the data follows a normal distribution, Standardisation is generally the better choice. For non-Gaussian distributions, Normalisation might be more suitable.

- Algorithm Requirements: Some algorithms are more sensitive to the scale of data and benefit from one method over the other. For instance, neural networks can benefit from Normalisation, while linear models often perform better with standardised data.

- Impact on Performance: Ultimately, the choice can depend on empirical performance. It is often beneficial to try both methods and evaluate which one improves the performance of the model.

### Real-World Example

Imagine you're working on a project to predict house prices based on various features like the number of rooms, square footage, and age of the house. Here’s how you might decide between Normalisation and Standardisation:

#### Exploring Data Distribution

- The number of rooms might be uniformly distributed.
- Square footage could have a skewed distribution.
- Age might follow a normal distribution.

#### Algorithm Choice

- If you're using a linear regression model, Standardising the features could help the model converge faster and perform better.
- If you're training a neural network, normalizing the features might help in achieving faster convergence due to the bounded activation functions.

#### Practical Application

- normalise the square footage and number of rooms since they don't follow a normal distribution.
- Standardize the age of the house as it likely follows a normal distribution.

By understanding the nature of the data and the requirements of the model, the user can make an informed decision on whether to use Normalisation or Standardisation.
