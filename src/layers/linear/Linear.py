from ..Layer import Layer
import numpy as np
from differentiation import Node
from layers.initializers import getInitializer
from ..Parameters import Parameters



class Linear(Layer):
    def __init__(self, in_dim: int, out_dim: int, initializer: str="uniform"):
        super().__init__()
        initializer_fn = getInitializer(initializer)
        self.weights = Parameters( Node(initializer_fn(in_dim, out_dim)) )
        self.biases = Parameters( Node(np.zeros((out_dim, 1))) )


    def __call__(self, batch: list[Node]) -> list[Node]:
        return [
            (self.weights.node.T @ x) + self.biases.node
            for x in batch
        ]
    

    def zero_grad(self):
        self.weights.zero_grad()
        self.biases.zero_grad()

    
    def parameters(self) -> list[Parameters]:
        return [ self.weights, self.biases ]