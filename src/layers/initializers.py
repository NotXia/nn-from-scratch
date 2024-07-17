import numpy as np
from typing import Callable



def getInitializer(name: str) -> Callable:
    match name:
        case "xavier": return xavier


def xavier(in_dim: int, out_dim: int) -> np.ndarray:
    return np.random.uniform(-1 / np.sqrt(in_dim), 1 / np.sqrt(in_dim), (in_dim, out_dim))