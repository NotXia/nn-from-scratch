from .Optimizer import Optimizer
from layers import Parameters



class SGD(Optimizer):
    def __init__(self, parameters: list[Parameters], lr: float=1e-3):
        super().__init__(parameters, lr)
            
    
    def step(self):
        for param in self.parameters:
            param.step( self.lr * param.grad )