from abc import ABC, abstractmethod
from differentiation import Node



class Optimizer(ABC):
    def __init__(self, parameters: list[Node], lr: float=1e-3):
        self.parameters = parameters
        self.lr = lr

    
    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()
            
    
    @abstractmethod
    def step(self):
        pass