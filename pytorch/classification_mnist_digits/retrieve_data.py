import random
import numpy as np

import torch

from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader


SEED_VALUE = 42
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)
torch.mps.manual_seed(SEED_VALUE)


def getData(rootDir, batchSize, numWorkers=1):
    preprocessTransforms = T.Compose(
        [
            # Convert (H,W,C) to (C,H,W) and normalise data to [0., 1.] by dividing by 255.
            T.ToTensor(),
            # Subtract mean(0.1307) and divide by variance(0.3081).
            #  This mean and variance is calculated on the training data.
            T.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )

    # Training set
    trainDataset = datasets.MNIST(
        rootDir, train=True, download=True, transform=preprocessTransforms
    )

    trainDataLoader = DataLoader(
        dataset=trainDataset,
        batch_size=batchSize,
        num_workers=numWorkers,
        shuffle=True,
    )

    # Validation set
    validationDataset = datasets.MNIST(
        rootDir, train=False, transform=preprocessTransforms
    )

    validationDataLoader = DataLoader(
        validationDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
    )

    return trainDataLoader, validationDataLoader
