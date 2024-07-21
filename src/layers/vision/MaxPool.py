from ..Layer import Layer
import numpy as np
from differentiation import Node
from ..Parameters import Parameters
from .utils import getOutputShape



class MaxPool(Layer):
    def __init__(self, 
        in_channels: int,
        kernel_size: int = 2, 
        stride: int = 2
    ):
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride


    def __call__(self, batch: list[Node]|list[np.ndarray]) -> list[Node]:
        if not isinstance(batch[0], Node): batch = [ Node(x) for x in batch ]
        assert all([ x.value.shape == batch[0].value.shape for x in batch ])

        in_width, in_height, in_channels = batch[0].value.shape
        out_width, out_height = getOutputShape(in_width, in_height, self.kernel_size, self.kernel_size, self.stride)
        batch_activations = []

        for x in batch:
            mask = np.zeros((in_width, in_height, in_channels))

            # Builds a mask with 1 at the positions of the max values
            for ch in range(in_channels):
                for j in range(out_height):
                    for i in range(out_width):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        kernel_slice = ( slice(start_i, start_i+self.kernel_size), slice(start_j, start_j+self.kernel_size), slice(ch, ch+1) )
                        max_idx = np.argmax(x[ kernel_slice ].value)
                        mask[kernel_slice][max_idx // self.kernel_size, max_idx % self.kernel_size] = 1

            batch_activations.append( x.maskAndShrink(mask, (out_width, out_height, in_channels)) )

        return batch_activations
    

    def zero_grad(self):
        pass

    
    def parameters(self) -> list[Parameters]:
        return []