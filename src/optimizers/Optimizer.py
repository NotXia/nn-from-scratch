from abc import ABC, abstractmethod
from layers import Parameters



class Optimizer(ABC):
    def __init__(self, parameters: list[Parameters], lr: float=1e-3):
        self.parameters = parameters
        self.lr = lr

    
    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()
            
    
    @abstractmethod
    def step(self):
        pass