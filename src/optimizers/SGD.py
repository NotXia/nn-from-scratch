from .Optimizer import Optimizer
from differentiation import Node



class SGD(Optimizer):
    def __init__(self, parameters: list[Node], lr: float=1e-3):
        super().__init__(parameters, lr)
            
    
    def step(self):
        for param in self.parameters:
            param.value -= self.lr * param.grad