from ..Layer import Layer
import numpy as np
from differentiation import Node
from ..Parameters import Parameters



class BatchNorm2D(Layer):
    def __init__(self, in_channels: int, momentum: float=0.1, eps: float=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.momentum = momentum
        self.eps = eps
        self.norm_weight = Parameters( Node(1) )
        self.bias = Parameters( Node(0) )
        self.running_mean = 0.0
        self.running_var = 0.0


    def __call__(self, batch: list[Node]) -> list[Node]:
        if not isinstance(batch[0], Node): batch = [ Node(x) for x in batch ]
        assert all([ x.shape == batch[0].shape for x in batch ])
        assert batch[0].shape[0] == self.in_channels
                
        if self.mode == "train":
            ch_means = np.mean([x.value for x in batch], axis=(0, 2, 3))
            ch_variances = np.var([x.value for x in batch], axis=(0, 2, 3))

            mean = np.zeros_like(batch[0].value)
            variance = np.zeros_like(batch[0].value)
            for ch in range(self.in_channels): 
                mean[ch] = ch_means[ch]
                variance[ch] = ch_variances[ch]

            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean
            self.running_var = (1-self.momentum)*self.running_var + self.momentum*variance
        else:
            mean = self.running_mean
            variance = self.running_var

        norm_activations = [
            ((x - mean) / ((variance + self.eps)**(1/2)))*self.norm_weight.node + self.bias.node
            for x in batch
        ]

        return norm_activations
    

    def zero_grad(self):
        self.norm_weight.zero_grad()
        self.bias.zero_grad()

    
    def parameters(self) -> list[Parameters]:
        return [ self.norm_weight, self.bias ]
