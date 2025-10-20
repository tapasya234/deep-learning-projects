import os
import math
import numpy as np
from numpy import loadtxt  # Function to load data from a text file
import pandas as pd  # Provides high-performance, easy-to-use data structures and data analysis tools
import seaborn as sns  # Provides high-level interface for drawing attractive statisticalgraphics
import matplotlib.pyplot as plt
import ucimlrepo as uci

# Fetch dataset
autoMPG = uci.fetch_ucirepo(id=9)
columnsNames = autoMPG.variables.name
# print(columnsNames.shape)

dataset = pd.concat([autoMPG.data.features, autoMPG.data.targets], axis=1)
# print(dataset.shape)
# print(dataset.head)

# Find the missing values in each column and drop them from the dataset.
# print(dataset.isna().sum())
# print(dataset.info())
dataset = dataset.dropna()
# print(dataset.shape)
# print(dataset.isna().sum())
# print(dataset.info())

# Explore the dataset
plt.figure(figsize=(12, 8))

# Generate a heatmap of the correlation matrix.
# sns.heatmap(dataset.corr(), cmap=plt.cm.Greens, annot=True)
# plt.title("Relationship between the features of the data", fontsize=13)

# Generate a pairplot for selected features (MPG, Horsepower, Displacement, Weight).
# diag_kind = "kde" - Kernel Density Estimation
# sns.pairplot(dataset, diag_kind="kde")
# sns.pairplot(dataset[["mpg", "horsepower", "displacement", "weight"]], diag_kind="kde")
# plt.show()


# Split the dataset into train and test
trainDataset = dataset.sample(frac=0.8, random_state=28)
testDataset = dataset.drop(trainDataset.index)

print(trainDataset.shape)
print(testDataset.shape)
# print(trainDataset)

# It is important to observe the statistics related to the dataset.
# When the feature data varies so widley, it is generally advised to scale the
# feature data as a pre-processing step before training a model.
# print(dataset.describe().transpose()[["mean", "std"]])

# Input Features and Target
X_train = trainDataset.copy()
X_test = testDataset.copy()

Y_train = X_train.pop("mpg")
Y_test = X_test.pop("mpg")

X_train_stats = X_train.describe().transpose()[["mean", "std"]]
print(X_train_stats)

# Normalize features
mean = X_train["horsepower"].mean()
std = X_train["horsepower"].std()
X_train["horsepower_standardised"] = (X_train["horsepower"] - mean) / std

print("Horsepower (Before) Mean: {} Std: {}", mean, std)
print(
    "Horsepower (After) Mean: {} Std: {}",
    X_train["horsepower_standardised"].mean(),
    X_train["horsepower_standardised"].std(),
)
