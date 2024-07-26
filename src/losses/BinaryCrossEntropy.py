from .Loss import Loss
from differentiation import Node
import numpy as np



class BinaryCrossEntropy(Loss):
    def __init__(self):
        pass


    def __call__(self, preds: list[Node], labels: np.ndarray) -> Node:
        batch_loss = Node(0)
        
        for pred, label in zip(preds, labels):
            batch_loss += -(label * Node.log(pred) + (1-label) * Node.log(1-pred))
        batch_loss = batch_loss / len(preds)
        batch_loss = batch_loss.sum()

        return batch_loss