from .Layer import Layer
from differentiation import Node



class Sequential(Layer):
    def __init__(self, layers: list[Layer]):
        self.layers = layers


    def __call__(self, x: Node):
        for l in self.layers:
            x = l(x)
        return x
    

    def zero_grad(self):
        for l in self.layers:
            l.zero_grad()

    
    def parameters(self):
        params = []
        for l in self.layers:
            params += l.parameters()
        return params