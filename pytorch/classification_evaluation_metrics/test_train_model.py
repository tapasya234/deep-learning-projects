import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from trainer import Trainer
from logistic_regression_model import LogisticRegressionModel

NUM_CLASSES = 2

seed = 42
rng = np.random.RandomState(seed)
torch.manual_seed(seed)
torch.mps.manual_seed(seed)


def generateDatasets():
    # Generate two class classification problem
    X, y = make_classification(
        n_features=NUM_CLASSES,
        n_redundant=0,
        n_informative=2,
        random_state=seed,
        n_clusters_per_class=1,
    )

    X += 4 * rng.uniform(size=X.shape)

    print(f"Inputs (X) shape: {X.shape} ")
    print(f"Inputs (y) shape: {y.shape} ")

    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    plt.grid("on")
    plt.show()

    return X, y


def toTorch(array) -> torch.Tensor:
    return torch.from_numpy(array).float()


def splitDataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )

    return toTorch(X_train), toTorch(X_test), toTorch(y_train), toTorch(y_test)


def getPredictions():
    X, y = generateDatasets()
    X_train, X_test, y_train, y_test = splitDataset(X, y)

    model = LogisticRegressionModel(nFeatures=NUM_CLASSES)

    # Use Binary Cross-Entropy Loss
    criterion = nn.BCELoss()
    # Use Stochastic Gradient Descent
    optimiser = optim.SGD(model.parameters(), lr=0.01)

    trainer = Trainer(model, criterion, optimiser, 200)

    y_train.unsqueeze_(1)
    trainer.fit(X_train, y_train)

    return trainer.predict(X_test), y_test
