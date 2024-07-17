from .Loss import Loss
from differentiation import Node



class BinaryCrossEntropy(Loss):
    def __init__(self):
        pass


    def __call__(self, pred: Node, label: Node) -> Node:
        return -(label * Node.log(pred) + (1-label) * Node.log(1-pred))