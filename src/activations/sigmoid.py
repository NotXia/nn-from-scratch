from differentiation import Node

def sigmoid(x: Node|list[Node]) -> Node:
    if isinstance(x, Node):
        return 1.0 / (1.0 + Node.exp(-x))
    else:
        return [ sigmoid(a) for a in x ]