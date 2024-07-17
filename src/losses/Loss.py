from abc import ABC, abstractmethod
from differentiation import Node



class Loss(ABC):
    @abstractmethod
    def __call__(self, pred: Node, label: Node):
        pass