from .Layer import Layer
from differentiation import Node
from activations import *



class _ActivationLayer(Layer):
    def __init__(self, activation_fn):
        self.activation_fn = activation_fn

    def __call__(self, batch: list[Node]):
        return [ self.activation_fn(x) for x in batch ]

    def zero_grad(self):
        pass

    def parameters(self):
        return []
    

class Sigmoid(_ActivationLayer):
    def __init__(self): super().__init__(sigmoid)

class Tanh(_ActivationLayer):
    def __init__(self): super().__init__(tanh)

class ReLU(_ActivationLayer):
    def __init__(self): super().__init__(relu)

class Softmax(_ActivationLayer):
    def __init__(self): super().__init__(softmax)