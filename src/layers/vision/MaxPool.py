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


    def __buildConvMatrix(self, input: np.ndarray):
        in_channels, in_height, in_width = input.shape
        out_width, out_height = getOutputShape(in_width, in_height, self.kernel_size, self.kernel_size, self.stride)
        in_flattened_shape = in_width * in_height * in_channels
        out_activations = out_width * out_height * in_channels
        conv_matr = np.zeros(( out_activations, in_flattened_shape ))

        for ch in range(in_channels):
            for i in range(out_height):
                for j in range(out_width):
                    start_i = i * self.stride
                    start_j = j * self.stride
                    kernel_slice = ( slice(ch, ch+1), slice(start_i, start_i+self.kernel_size), slice(start_j, start_j+self.kernel_size) )
                    max_kernel_idx = np.argmax(input[ kernel_slice ])
                    max_idx = ch*(in_width*in_height) + ( (start_i + max_kernel_idx//self.kernel_size)*in_width + (start_j + max_kernel_idx%self.kernel_size) )
                    activation_idx = ch*(out_width*out_height) + (i*out_width + j)
                    conv_matr[ activation_idx, max_idx ] = 1

        return Node(conv_matr)
    

    def __call__(self, batch: list[Node]|list[np.ndarray]) -> list[Node]:
        if not isinstance(batch[0], Node): batch = [ Node(x) for x in batch ]
        assert all([ x.value.shape == batch[0].value.shape for x in batch ])

        in_channels, in_height, in_width = batch[0].value.shape
        out_width, out_height = getOutputShape(in_width, in_height, self.kernel_size, self.kernel_size, self.stride)
        batch_activations = []

        for x in batch:
            conv_matrix = self.__buildConvMatrix(x.value)
            x = x.flatten().reshape((-1, 1))
            batch_activations.append( (conv_matrix @ x).reshape((in_channels, out_height, out_width)) )

        return batch_activations
    

    def zero_grad(self):
        pass

    
    def parameters(self) -> list[Parameters]:
        return []