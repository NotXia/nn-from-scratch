from .Layer import Layer
import numpy as np
from differentiation import Node
from layers.initializers import getInitializer



class Linear(Layer):
    def __init__(self, in_dim: int, out_dim: int, initializer: str="xavier"):
        initializer_fn = getInitializer(initializer)
        self.weights = Node( initializer_fn(in_dim, out_dim) )
        self.biases = Node( np.zeros((out_dim, )) )


    def __call__(self, x: Node):
        return (x @ self.weights) + self.biases
    

    def zero_grad(self):
        self.weights.zero_grad()
        self.biases.zero_grad()

    
    def parameters(self):
        return [ self.weights, self.biases ]