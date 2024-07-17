from .Optimizer import Optimizer
from differentiation import Node
import numpy as np



class Adam(Optimizer):
    def __init__(self, parameters: list[Node], lr: float=1e-3, 
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ):
        super().__init__(parameters, lr)
        self.tot_steps = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.moment1 = []
        self.moment2 = []
        for param in self.parameters:
            self.moment1.append( np.zeros_like(param.value) )
            self.moment2.append( np.zeros_like(param.value) )
            
    
    def step(self):
        self.tot_steps += 1

        for i, param in enumerate(self.parameters):
            grad = param.grad
            self.moment1[i] = self.beta1*self.moment1[i] + (1-self.beta1)*grad
            self.moment2[i] = self.beta2*self.moment2[i] + (1-self.beta2)*grad**2
            m = self.moment1[i] / (1 - self.beta1**self.tot_steps)
            v = self.moment2[i] / (1 - self.beta2**self.tot_steps)
            param.value -= (self.lr * m) / (np.sqrt(v)+ self.eps)