# utils/seed_utils.py
# BharatWitness deterministic seeding utilities

import random
import numpy as np
import torch
from typing import Optional


def set_deterministic_seed(seed: Optional[int] = None) -> int:
    if seed is None:
        seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed
