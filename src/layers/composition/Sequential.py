from ..Layer import Layer
from differentiation import Node



class Sequential(Layer):
    def __init__(self, layers: list[Layer]):
        super().__init__()
        self.layers = layers


    def __call__(self, batch: list[Node]):
        for l in self.layers:
            batch = l(batch)
        return batch
    

    def zero_grad(self):
        for l in self.layers:
            l.zero_grad()

    
    def parameters(self):
        params = []
        for l in self.layers:
            params += l.parameters()
        return params
    
    
    def train(self):
        for l in self.layers:
            l.train()

    def inference(self):
        for l in self.layers:
            l.inference()