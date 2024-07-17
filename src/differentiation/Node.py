"""
    Node of the computational graph used for reverse automatic differentiation.
"""
from __future__ import annotations
import numpy as np



def _matchShape(arr: np.ndarray, target_shape: tuple[int]) -> np.ndarray:
    # Match number of dimensions
    arr = arr.sum(axis=tuple(range(arr.ndim - len(target_shape))))

    # Match dimensions shape
    for i, dim_shape in enumerate(target_shape):
        if dim_shape == 1:
            arr = np.sum(arr, axis=i, keepdims=True)

    return arr


class Node:
    @staticmethod
    def log(node: Node): 
        return node.log()
        
    @staticmethod
    def exp(node: Node): 
        return node.exp()


    def __init__(self, value: np.ndarray|float, parents: list[Node]|None=None):
        self.value = np.array(value, dtype=float)
        self.parents = parents if parents is not None else []
        self.zero_grad()


    def __str__(self):
        return f"{self.value} (grad: {self.grad})"

    
    def zero_grad(self):
        self.grad = np.zeros_like(self.value)


    def storeDerivatives(self):
        pass


    def backward(self):
        """
            Traverses the graph backwards in topological order
            and computes the derivatives.
        """
        self.grad = np.ones_like(self.value)
        nodes_to_grad = [ self ]

        while len(nodes_to_grad) > 0:
            node = nodes_to_grad.pop(0)
            node.storeDerivatives()
            nodes_to_grad += node.parents


    def __add__(self, node: Node|float) -> Node:
        return _Addition(self, node)
    
    def __radd__(self, node: Node|float) -> Node:
        return _Addition(node, self)
    

    def __sub__(self, node: Node|float) -> Node:
        return _Addition(self, _Negation(node))
    
    def __rsub__(self, node: Node|float) -> Node:
        return _Addition(node, _Negation(self))


    def __neg__(self) -> Node:
        return _Negation(self)
    

    def __mul__(self, node: Node|float) -> Node:
        return _Product(self, node)

    def __rmul__(self, node: Node|float) -> Node:
        return _Product(node, self)
    

    def __truediv__(self, node: Node|float) -> Node:
        return _Division(self, node)

    def __rtruediv__(self, node: Node|float) -> Node:
        return _Division(node, self)
    

    def __pow__(self, node: Node|float) -> Node:
        return _Power(self, node)

    def __rpow__(self, node: Node|float) -> Node:
        return _Power(node, self)
    

    def log(self) -> Node:
        return _NaturalLog(self)
    
    def exp(self) -> Node:
        return _Exponential(self)


    def __matmul__(self, node: Node|float) -> Node:
        return _MatrixMultiplication(self, node)
    
    def __rmatmul__(self, node: Node|float) -> Node:
        return _MatrixMultiplication(node, self)
    


class _Addition(Node):
    def __init__(self, node1: Node|float, node2: Node|float):
        if not isinstance(node1, Node): node1 = Node([node1])
        if not isinstance(node2, Node): node2 = Node([node2])
        super().__init__(node1.value + node2.value, [node1, node2])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad * 1, self.parents[0].grad.shape )
        self.parents[1].grad += _matchShape( self.grad * 1, self.parents[1].grad.shape )


class _Negation(Node):
    def __init__(self, node: Node|float):
        if not isinstance(node, Node): node = Node([node])
        super().__init__(-node.value, [node])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad * -1, self.parents[0].grad.shape )


class _Product(Node):
    def __init__(self, node1: Node|float, node2: Node|float):
        if not isinstance(node1, Node): node1 = Node([node1])
        if not isinstance(node2, Node): node2 = Node([node2])
        super().__init__(node1.value * node2.value, [node1, node2])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad * self.parents[1].value, self.parents[0].grad.shape )
        self.parents[1].grad += _matchShape( self.grad * self.parents[0].value, self.parents[1].grad.shape )


class _Division(Node):
    def __init__(self, node1: Node|float, node2: Node|float):
        if not isinstance(node1, Node): node1 = Node([node1])
        if not isinstance(node2, Node): node2 = Node([node2])
        super().__init__(node1.value / node2.value, [node1, node2])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad / self.parents[1].value, self.parents[0].grad.shape )
        self.parents[1].grad += _matchShape( self.grad * (-self.parents[0].value / (self.parents[1].value**2)), self.parents[1].grad.shape )


class _Power(Node):
    def __init__(self, base_node: Node|float, exp_node: Node|float):
        if not isinstance(base_node, Node): base_node = Node([base_node])
        if not isinstance(exp_node, Node): exp_node = Node([exp_node])
        super().__init__(base_node.value ** exp_node.value, [base_node, exp_node])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad * self.parents[1].value * (self.parents[0].value**(self.parents[1].value-1)), self.parents[0].grad.shape)
        self.parents[1].grad += _matchShape( self.grad * (self.parents[0].value**self.parents[1].value) * np.log(self.parents[0].value), self.parents[1].grad.shape)


class _NaturalLog(Node):
    def __init__(self, node: Node|float):
        if not isinstance(node, Node): node = Node([node])
        super().__init__(np.log(node.value), [node])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad * (1/self.parents[0].value), self.parents[0].grad.shape )


class _Exponential(Node):
    def __init__(self, node: Node|float):
        if not isinstance(node, Node): node = Node([node])
        super().__init__(np.exp(node.value), [node])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad * np.exp(self.parents[0].value), self.parents[0].grad.shape )


class _MatrixMultiplication(Node):
    def __init__(self, node1: Node|float, node2: Node|float):
        if not isinstance(node1, Node): node1 = Node([node1])
        if not isinstance(node2, Node): node2 = Node([node2])
        super().__init__(node1.value @ node2.value, [node1, node2])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad @ self.parents[1].value.T, self.parents[0].grad.shape )
        self.parents[1].grad += _matchShape( self.parents[0].value.T @ self.grad, self.parents[1].grad.shape )
