from differentiation import Node

def tanh(x: Node) -> Node:
    return (Node.exp(x) - Node.exp(-x)) / (Node.exp(x) + Node.exp(-x))