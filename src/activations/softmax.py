from differentiation import Node

def softmax(x: Node|list[Node]) -> Node:
    if isinstance(x, Node):
        return Node.exp(x) / Node.exp(x).sum(axis=None)
    else:
        return [ softmax(a) for a in x ]