from ..Layer import Layer
import numpy as np
from differentiation import Node
from ..Parameters import Parameters



class Flatten(Layer):
    def __init__(self):
        super().__init__()


    def __call__(self, batch: list[Node]|list[np.ndarray]) -> list[Node]:
        if not isinstance(batch[0], Node): batch = [ Node(x) for x in batch ]
        assert all([ x.value.shape == batch[0].value.shape for x in batch ])

        return [x.flatten().reshape((-1, 1)) for x in batch]
    

    def zero_grad(self):
        pass

    
    def parameters(self) -> list[Parameters]:
        return []