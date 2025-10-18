import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms as T


def imagePreprocessTransforms(
    imgSize: Tuple[int] = (32, 32), mean: float = None, stdDev: float = None
):
    transformers = [
        # Resize image to specified size
        T.Resize(size=imgSize, antialias=True),
        # Update order of channels from (H,W,C) to (C,H,W) and
        # normalize data to [0., 1.] by dividing it by 255.
        T.ToTensor(),
    ]

    if mean is not None or stdDev is not None:
        transformers.append(
            # Subtract mean and divide by variance,
            # these values are calculated on the training data.
            T.Normalize(mean=mean, std=stdDev)
        )
    return T.Compose(transformers)


def getData(
    rootDir: str, imgSize: Tuple[int] = (32, 32)
) -> Tuple[datasets.FashionMNIST]:
    preprocessTransforms = imagePreprocessTransforms(imgSize=imgSize)

    shouldDownload = False if os.path.exists(rootDir) else True

    trainDataset = datasets.FashionMNIST(
        root=rootDir,
        train=True,
        transform=preprocessTransforms,
        download=shouldDownload,
    )

    mean = trainDataset.data.numpy().mean(axis=(0, 1, 2)) / 255
    stdDev = trainDataset.data.numpy().std(axis=(0, 1, 2)) / 255

    preprocessTransforms = imagePreprocessTransforms(
        imgSize=imgSize, mean=mean, stdDev=stdDev
    )
    trainDataset = datasets.FashionMNIST(
        root=rootDir,
        train=True,
        transform=preprocessTransforms,
        download=shouldDownload,
    )

    testDataset = datasets.FashionMNIST(
        root=rootDir,
        train=False,
        transform=preprocessTransforms,
        download=shouldDownload,
    )

    return trainDataset, testDataset


def createDataLoaders(
    trainDataset: datasets.FashionMNIST,
    testDataset: datasets.FashionMNIST,
    batchSize: int,
    numWorkers: int = 1,
    seed: int = 42,
) -> Tuple[DataLoader]:

    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)

    # Split training dataset into 90-10 training-validation dataset
    trainDataset, validationDataset = random_split(trainDataset, lengths=[0.9, 0.1])

    # generate data loaders
    trainDataLoader = DataLoader(
        dataset=trainDataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=numWorkers,
    )

    validationDataLoader = DataLoader(
        dataset=validationDataset,
        batch_size=batchSize,
        # shuffle=True,
        shuffle=False,
        num_workers=numWorkers,
    )

    testDataLoader = DataLoader(
        dataset=testDataset,
        batch_size=batchSize,
        # shuffle=True,
        shuffle=False,
        num_workers=numWorkers,
    )

    # Print stats
    print("Dataset Statistics: \n")
    print(
        f"Train: {len(trainDataset)} Validation: {len(validationDataset)} Test: {len(testDataset)}"
    )
    print(f"Image Shape: {trainDataset[0][0].shape}")

    return trainDataLoader, validationDataLoader, testDataLoader
