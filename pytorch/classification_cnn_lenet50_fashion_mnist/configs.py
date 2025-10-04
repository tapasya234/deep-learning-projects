from dataclasses import dataclass

from data_path import DATA_PATH


@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES: int = 10
    IMAGE_HEIGHT: int = 32
    IMAGE_WIDTH: int = 32
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 2
    DATA_ROOT: str = DATA_PATH + "FASHION_MNIST"


@dataclass(frozen=True)
class TrainingConfig:
    NUM_EPOCHS: int = 20
    LEARNING_RATE: float = 1e-2
    CHECKPOINT_DIR: str = DATA_PATH + "CHECKPOINTS"
    DEVICE: str = "mps"
