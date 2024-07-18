from .Loss import Loss
from differentiation import Node
import numpy as np



class CategoricalCrossEntropy(Loss):
    def __init__(self):
        pass


    def __call__(self, preds: list[Node], labels: list[Node]) -> Node:
        batch_loss = Node(0)
        
        for pred, label in zip(preds, labels):
            mask = np.zeros_like(pred.value)
            mask[np.argmax(label)] = 1
            batch_loss += -Node.sum( Node(mask) * Node.log(pred) )
        batch_loss = batch_loss / len(preds)

        return batch_loss