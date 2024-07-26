from abc import ABC, abstractmethod
from differentiation import Node
import numpy as np



class Loss(ABC):
    @abstractmethod
    def __call__(self, preds: list[Node], labels: np.ndarray) -> Node:
        pass