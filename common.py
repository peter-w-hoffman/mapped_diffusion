# File: common.py

import random
import numpy as np
import torch

IMG_SIZE = 28
D = IMG_SIZE * IMG_SIZE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device() -> None:
    print("device:", device)
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Seed set to:", seed)