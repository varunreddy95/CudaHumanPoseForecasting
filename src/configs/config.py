import os
import torch
from paths import DATASET_DIR
from dataclasses import dataclass


@dataclass
class DefaultConfig:

    HC_INIT_MODE: str = "random"
    BATCH_SIZE: int = 8
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_LR: float = 1e-3
    EPOCHS: int = 100
    GAMMA: float = 0.5
    STEPSIZE: int = 20


cfg = DefaultConfig()
