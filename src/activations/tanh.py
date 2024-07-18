from differentiation import Node

def tanh(x: Node|list[Node]) -> Node:    
    if isinstance(x, Node):
        return (Node.exp(x) - Node.exp(-x)) / (Node.exp(x) + Node.exp(-x))
    else:
        return [ tanh(a) for a in x ]