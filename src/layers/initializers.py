import numpy as np
from typing import Callable



def getInitializer(name: str) -> Callable:
    match name:
        case "xavier": return xavier
        case "he": return he
        case "zero": return zero
        case "uniform": return uniform


def getConvInitializer(name: str) -> Callable:
    match name:
        case "uniform": return uniformConv
        case "xavier": return xavierConv
        case "he": return heConv



def xavier(in_dim: int, out_dim: int) -> np.ndarray:
    return np.random.uniform(-1 / np.sqrt(in_dim), 1 / np.sqrt(in_dim), size=(in_dim, out_dim))


def he(in_dim: int, out_dim: int) -> np.ndarray:
    return np.random.normal(0.0, np.sqrt(2/in_dim), size=(in_dim, out_dim))


def zero(in_dim: int, out_dim: int) -> np.ndarray:
    return np.zeros((in_dim, out_dim))


def uniform(in_dim: int, out_dim: int) -> np.ndarray:
    # Same as in PyTorch
    bound = np.sqrt(1/in_dim)
    return np.random.uniform(-bound, bound, size=(in_dim, out_dim))



def xavierConv(channels: int, height: int, width: int) -> np.ndarray:
    in_dim = channels * height * width
    return np.random.uniform(-1 / np.sqrt(in_dim), 1 / np.sqrt(in_dim), size=(channels, height, width))


def heConv(channels: int, height: int, width: int) -> np.ndarray:
    in_dim = channels * height * width
    return np.random.normal(0.0, np.sqrt(2/in_dim), size=(channels, height, width))


def uniformConv(channels: int, height: int, width: int) -> np.ndarray:
    # Same as in PyTorch
    bound = np.sqrt( 1/(channels * height * width) )
    return np.random.uniform(-bound, bound, size=(channels, height, width))