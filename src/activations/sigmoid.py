from differentiation import Node

def sigmoid(x: Node) -> Node:
    return 1.0 / (1.0 + Node.exp(-x))