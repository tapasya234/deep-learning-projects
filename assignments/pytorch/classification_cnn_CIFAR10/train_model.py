from configs import TrainingConfig

import numpy as np
from torch import device
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train(
    DEVICE: device,
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    epoch_idx: int,
    log_interval: int,
) -> None:
    # change model in training mood
    model.train()

    # to get batch loss
    batch_loss = np.array([])

    # to get batch accuracy
    batch_acc = np.array([])

    for batch_idx, (data, target) in enumerate(train_loader):

        # clone target
        indx_target = target.clone()
        # send data to device (its is medatory if GPU has to be used)
        data = data.to(DEVICE)
        # send target to device
        target = target.to(DEVICE)

        # reset parameters gradient to zero
        optimizer.zero_grad()

        # forward pass to the model
        output = model(data)

        # cross entropy loss
        loss = F.cross_entropy(output, target)

        # find gradients w.r.t training parameters
        loss.backward()
        # Update parameters using gardients
        optimizer.step()

        batch_loss = np.append(batch_loss, [loss.item()])

        # Score to probability using softmax
        prob = F.softmax(output, dim=1)

        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]

        # correct prediction
        correct = pred.cpu().eq(indx_target).sum()

        # accuracy
        acc = float(correct) / float(len(data))

        batch_acc = np.append(batch_acc, [acc])

        if batch_idx % log_interval == 0 and batch_idx > 0:
            print(
                "Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f}".format(
                    epoch_idx,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    loss.item(),
                    acc,
                )
            )

    return batch_loss.mean(), batch_acc.mean()
