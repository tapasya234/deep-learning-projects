import os
import torch
from torchvision import datasets, transforms
import numpy as np

IMG_SIZE = (32, 32)


def getMeanStdDev(
    dataRoot: str, imgWidth: int, imgHeight: int, shouldDownload: bool = False
):

    train_transform = transforms.Compose(
        [
            # transforms.Resize(size=(imgHeight, imgWidth)),
            transforms.ToTensor(),
        ]
    )
    train_set = datasets.CIFAR10(
        root=dataRoot, train=True, download=shouldDownload, transform=train_transform
    )

    # return mean (numpy.ndarray) and std (numpy.ndarray)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    ###
    ### YOUR CODE HERE
    ###
    mean = train_set.data.mean(axis=(0, 1, 2)) / 255
    std = train_set.data.std(axis=(0, 1, 2)) / 255

    return mean, std


def getData(
    batchSize: int,
    dataRoot: str,
    imgWidth: int,
    imgHeight: int,
    numWorkers: int = 1,
):

    print(dataRoot)
    shouldDownload = not os.path.exists(dataRoot)
    if shouldDownload:
        os.makedirs(dataRoot)

    try:
        mean, std = getMeanStdDev(
            dataRoot,
            imgWidth=imgWidth,
            imgHeight=imgHeight,
            shouldDownload=shouldDownload,
        )
        assert len(mean) == len(std) == 3
    except:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])

    print(f"Mean: {mean} StdDev: {std}")
    train_test_transforms = transforms.Compose(
        [
            # this re-scale image tensor values between 0-1. image_tensor /= 255
            transforms.ToTensor(),
            # subtract mean and divide by variance.
            transforms.Normalize(mean, std),
        ]
    )

    # train dataloader
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=dataRoot,
            train=True,
            download=shouldDownload,
            transform=train_test_transforms,
        ),
        batch_size=batchSize,
        shuffle=True,
        num_workers=numWorkers,
        pin_memory=True,
    )

    # test dataloader
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=dataRoot,
            train=False,
            download=shouldDownload,
            transform=train_test_transforms,
        ),
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=True,
    )
    return train_loader, test_loader
