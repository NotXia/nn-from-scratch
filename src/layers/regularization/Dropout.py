from ..Layer import Layer
import numpy as np
from differentiation import Node
from ..Parameters import Parameters



class Dropout(Layer):
    def __init__(self, drop_prob: float=0.1):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1-drop_prob


    def __call__(self, batch: list[Node]) -> list[Node]:
        if not isinstance(batch[0], Node): batch = [ Node(x) for x in batch ]
        assert all([ x.shape == batch[0].shape for x in batch ])
        
        # Inverted dropout
        if self.mode == "train":
            mask = (np.random.rand(*batch[0].shape) <= self.keep_prob).astype(float) / self.keep_prob
            return [ x * mask for x in batch ]
        else:
            return batch
        

    def zero_grad(self):
        pass

    
    def parameters(self) -> list[Parameters]:
        return []