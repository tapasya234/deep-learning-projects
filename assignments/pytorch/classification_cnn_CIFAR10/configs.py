from dataclasses import dataclass

from data_path import DATA_PATH


@dataclass(frozen=True)
class DatasetConfig:
    # amount of data to pass through the network at each forward-backward iteration
    NUM_CLASSES: int = 10

    IMAGE_HEIGHT: int = 32
    IMAGE_WIDTH: int = 32
    BATCH_SIZE: int = 16
    # number of concurrent processes using to prepare data
    NUM_WORKERS: int = 4
    # folder to save data
    DATA_ROOT: str = DATA_PATH + "DATA"


@dataclass(frozen=True)
class TrainingConfig:
    """
    Describes configuration of the training process
    """

    # number of times the whole dataset will be passed through the network
    NUM_EPOCHS: int = 20
    # determines the speed of network's weights update
    LEARNING_RATE: float = 0.001
    # Folder to store the saved model
    MODELS_DIR: str = DATA_PATH + "models"
    # Name of the saved model
    MODEL_NAME = "cifar10_cnn_model.pt"
    # device to use for training.
    DEVICE: str = "mps"
    # how many batches to wait between logging training status
    LOG_INTERVAL: int = 500
    # how many epochs to wait before another test. Set to 1 to get val loss at each epoch
    TEST_INTERVAL: int = 1


@dataclass
class SystemConfig:
    """
    Describes the common system setting needed for reproducible training
    """

    # seed number to set the state of all random number generators
    SEED: int = 42
