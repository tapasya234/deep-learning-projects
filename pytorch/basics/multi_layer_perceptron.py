import os
import random

# import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchinfo import summary

# Environment variable that allows PyTorch to fall back to CPU execution
# when encountering operations that are not currently supported by MPS.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
torch.use_deterministic_algorithms(True)

SEED_VALUE = 42
random.seed(SEED_VALUE)
# np.random.seed(SEED_VALUE)
torch.mps.manual_seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)


class MLP(nn.Module):
    def __init__(self, numInputs, numHiddenLayerNodes, numOutputs):
        super().__init__()

        self.linear1 = nn.Linear(
            in_features=numInputs, out_features=numHiddenLayerNodes
        )
        self.linear2 = nn.Linear(
            in_features=numHiddenLayerNodes, out_features=numOutputs
        )

    def forward(self, x):
        # Forward pass through hidden layer
        x = F.relu(self.linear1(x))

        # Forward pass to output layer
        return self.linear2(x)


class MLP_Sequential(nn.Module):
    def __init__(self, numInputs, numHiddenLayerNodes, numOutputs):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(numInputs, numHiddenLayerNodes),
            nn.ReLU(),
            nn.Linear(numHiddenLayerNodes, numOutputs),
        )

    def forward(self, x):
        return self.model(x)


def trainModel(
    model,
    lossFunction: nn.MSELoss,
    optimiser: optim.SGD,
    data: torch.Tensor,
    targets: torch.Tensor,
    numEpochs,
):
    model.train()

    for epochIndex in range(numEpochs):
        optimiser.zero_grad()

        yPredictions = model(data)

        loss = lossFunction(yPredictions, targets)
        loss.backward()
        optimiser.step()

    print(f"Loss: {loss.detach().item():.4f}")
    return


numData = 1000
numInputs = 1000
numOutputs = 10
numEpochs = 100
numHiddenNodes = 100

torch.manual_seed(SEED_VALUE)
# Create random Tensors to hold inputs and outputs
X = torch.randn(numData, numInputs)
y = torch.randn(numData, numOutputs)

print("--------------")
print("MLP Model")
model = MLP(
    numInputs=numInputs,
    numHiddenLayerNodes=numHiddenNodes,
    numOutputs=numOutputs,
)
print(
    summary(model, input_size=(1, numInputs), device="cpu", row_settings=["var_names"])
)
trainModel(
    model=model,
    lossFunction=nn.MSELoss(reduction="sum"),
    optimiser=optim.SGD(model.parameters(), lr=1e-4),
    data=X,
    targets=y,
    numEpochs=numEpochs,
)

print("--------------")
print("MLP_Sequential Model")
modelSequential = MLP_Sequential(
    numInputs=numInputs,
    numHiddenLayerNodes=numHiddenNodes,
    numOutputs=numOutputs,
)
print(
    summary(model, input_size=(1, numInputs), device="cpu", row_settings=["var_names"])
)
trainModel(
    model=modelSequential,
    lossFunction=nn.MSELoss(reduction="sum"),
    optimiser=optim.SGD(modelSequential.parameters(), lr=1e-4),
    data=X,
    targets=y,
    numEpochs=numEpochs,
)
