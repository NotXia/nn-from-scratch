from .Loss import Loss
from differentiation import Node
import numpy as np



class CategoricalCrossEntropy(Loss):
    def __init__(self, label_smoothing: float|None=None):
        self.label_smoothing = label_smoothing


    def __call__(self, preds: list[Node], labels: list[Node]) -> Node:
        batch_loss = Node(0)
        
        for pred, label in zip(preds, labels):
            label = np.expand_dims(label, axis=-1)
            if self.label_smoothing is not None:
                smooth_factor = self.label_smoothing / len(label)
                label[label == 0] = smooth_factor
                label[label == 1] = 1 - smooth_factor*(len(label)-1)

            batch_loss += -Node.sum( Node(label) * Node.log(pred) )
        batch_loss = batch_loss / len(preds)

        return batch_loss