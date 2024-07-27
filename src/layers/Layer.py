from abc import ABC, abstractmethod
from differentiation import Node
from .Parameters import Parameters



class Layer(ABC):
    def __init__(self):
        self.mode = "train"


    @abstractmethod
    def __call__(self, batch: list[Node]) -> list[Node]:
        pass
    

    @abstractmethod
    def zero_grad(self):
        pass

    
    @abstractmethod
    def parameters(self) -> list[Parameters]:
        pass


    def train(self):
        self.mode = "train"

    def inference(self):
        self.mode = "inference"