from abc import ABC
from ..Layer import Layer



class Module(Layer, ABC):
    def zero_grad(self):
        for var in vars(self).values():
            if isinstance(var, Layer):
                var.zero_grad()

    
    def parameters(self):
        params = []
        for var in vars(self).values():
            if isinstance(var, Layer):
                params += var.parameters()
        return params
    

    def train(self):
        for var in vars(self).values():
            if isinstance(var, Layer):
                var.train()

    def inference(self):
        for var in vars(self).values():
            if isinstance(var, Layer):
                var.inference()