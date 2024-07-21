from ..Layer import Layer
import numpy as np
from differentiation import Node
from layers.initializers import getConvInitializer
from ..Parameters import Parameters
from .utils import getOutputShape



class ParametersConvolution(Parameters):
    def __init__(self, kernel: np.ndarray, stride: int):
        self.node = None
        self.kernel = kernel
        self.channels, self.height, self.width = kernel.shape
        self.stride = stride
        self.params_position = {} # Keeps track of the positions of the kernel parameters in the convolution matrix


    def buildConvMatrix(self, in_width, in_height):
        in_flattened_shape = in_width * in_height * self.channels
        out_width, out_height = getOutputShape(in_width, in_height, self.width, self.height, self.stride)
        out_activations = out_width * out_height
        conv_matr = np.zeros(( out_activations, in_flattened_shape ))
        self.params_position = {}

        curr_activation = 0
        for i in range(out_height):
            for j in range(out_width):
                start_offset = (i*in_width) + (j*self.stride) # Offset before the start of the kernel
                inbetween_offset = ( # Offset between kernel rows (i.e. skips to the position where the next row of the kernel starts)
                    (in_width - (start_offset + self.width)) + # Offset to kernel right side
                    start_offset                               # Offset to kernel left side
                )

                for k_c in range(self.channels):
                    for k_h in range(self.height):
                        for k_w in range(self.width):
                            conv_matr_row = curr_activation
                            conv_matr_col = start_offset + (k_w + k_h*self.width) + (k_h*inbetween_offset) + (k_c*in_width*in_height)
                            conv_matr[curr_activation, conv_matr_col] = self.kernel[k_c, k_h, k_w]

                            if (k_c, k_h, k_w) not in self.params_position: self.params_position[(k_c, k_h, k_w)] = []
                            self.params_position[(k_c, k_h, k_w)].append( (conv_matr_row, conv_matr_col) )

                curr_activation += 1

        self.node = Node(conv_matr)
        return self.node


    def zero_grad(self):
        if self.node is not None:
            self.node.zero_grad()


    @property
    def grad(self):
        # The gradient of a parameter of the kernel is obtained as the mean 
        # of the gradients corresponding to that parameter in the convolution matrix.
        kernel_grad = np.zeros_like(self.kernel)
        for c, h, w in self.params_position.keys():
            for conv_matr_row, conv_matr_col in self.params_position[(c, h, w)]:
                kernel_grad[c, h, w] += self.node.grad[conv_matr_row, conv_matr_col]
            kernel_grad[c, h, w] /= len(self.params_position[(c, h, w)])
        return kernel_grad


    @property
    def value(self):
        return self.kernel
    

    def step(self, step_size):
        self.kernel -= step_size 



class Conv2D(Layer):
    def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: str="valid", 
        initializer: str="uniform"
    ):
        initializer_fn = getConvInitializer(initializer)
        self.kernels = [ ParametersConvolution(initializer_fn(in_channels, kernel_size, kernel_size), stride) for _ in range(out_channels) ]
        self.biases = [ Parameters(Node([[0]])) for _ in range(out_channels) ]
        match padding:
            case "valid": self.padding = 0
            case "same":  self.padding = kernel_size // 2


    def __call__(self, batch: list[Node]|list[np.ndarray]) -> list[Node]:
        if not isinstance(batch[0], Node): batch = [ Node(x) for x in batch ]
        assert all([ x.value.shape == batch[0].value.shape for x in batch ])

        # Pad inputs
        if self.padding != 0:
            for i in range(len(batch)):
                batch[i] = batch[i].pad(((self.padding,), (self.padding,), (0,)))

        in_width, in_height, in_channels = batch[0].value.shape
        batch_activations = []

        for kernel in self.kernels:
            kernel.buildConvMatrix(in_width, in_height)

        for x in batch:
            activations = []
            x = x.flatten().reshape((-1, 1))
            for kernel, bias in zip(self.kernels, self.biases):
                out_width, out_height = getOutputShape(in_width, in_height, kernel.width, kernel.height, kernel.stride)
                activations.append( (kernel.node @ x + bias.node).reshape((out_width, out_height, 1)) )
            batch_activations.append( Node.concatenate(activations, axis=-1) )

        return batch_activations
    

    def zero_grad(self):
        for kernel in self.kernels: kernel.zero_grad()
        for bias in self.biases: bias.zero_grad()

    
    def parameters(self) -> list[Parameters]:
        return self.kernels + self.biases