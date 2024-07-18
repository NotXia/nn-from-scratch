from differentiation import Node, ReLU

def relu(x: Node|list[Node]) -> Node:    
    if isinstance(x, Node):
        return ReLU(x)
    else:
        return [ relu(a) for a in x ]