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


def toNode(x: Node|np.ndarray|float) -> Node:
    if isinstance(x, Node): 
        return x
    elif isinstance(x, float): 
        return Node([x])
    else:
        return Node(x) 


class Node:
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


    def __add__(self, node: Node|np.ndarray|float) -> Node:
        return _Addition(self, node)
    
    def __radd__(self, node: Node|np.ndarray|float) -> Node:
        return _Addition(node, self)
    

    def __sub__(self, node: Node|np.ndarray|float) -> Node:
        return _Addition(self, _Negation(node))
    
    def __rsub__(self, node: Node|np.ndarray|float) -> Node:
        return _Addition(node, _Negation(self))


    def __neg__(self) -> Node:
        return _Negation(self)
    

    def __mul__(self, node: Node|np.ndarray|float) -> Node:
        return _Product(self, node)

    def __rmul__(self, node: Node|np.ndarray|float) -> Node:
        return _Product(node, self)
    

    def __truediv__(self, node: Node|np.ndarray|float) -> Node:
        return _Division(self, node)

    def __rtruediv__(self, node: Node|np.ndarray|float) -> Node:
        return _Division(node, self)
    

    def __pow__(self, node: Node|np.ndarray|float) -> Node:
        return _Power(self, node)

    def __rpow__(self, node: Node|np.ndarray|float) -> Node:
        return _Power(node, self)
    

    def log(self) -> Node:
        return _NaturalLog(self)
    
    def exp(self) -> Node:
        return _Exponential(self)


    def __matmul__(self, node: Node|np.ndarray|float) -> Node:
        return _MatrixMultiplication(self, node)
    
    def __rmatmul__(self, node: Node|np.ndarray|float) -> Node:
        return _MatrixMultiplication(node, self)
    

    def transpose(self) -> Node:
        return _Transpose(self)

    @property
    def T(self):
        return self.transpose()


    def sum(self, axis: int|None=None, keepdims: bool=False) -> Node:
        return _Summation(self, axis, keepdims)
    

    def max(self) -> Node:
        return _Max(self)
    

    @staticmethod
    def concatenate(nodes: list[Node]|list[np.ndarray], axis: int=-1):
        return _Concatenate(nodes, axis)
    

    def flatten(self):
        return _Flatten(self)


    def reshape(self, new_shape: tuple[int]):
        return _Reshape(self, new_shape)
    

    def pad(self, pad_width: tuple[tuple[int]], mode="constant", **kwargs):
        return _Pad(self, pad_width, mode, **kwargs)
    

    def __getitem__(self, idx_slice: tuple[slice]):
        return _Slice(self, idx_slice)
    

    def maskAndShrink(self, mask: np.ndarray, new_shape: tuple[int]|None=None):
        return _MaskAndShrink(self, mask, new_shape)



class _Addition(Node):
    def __init__(self, node1: Node|np.ndarray|float, node2: Node|np.ndarray|float):
        node1 = toNode(node1)
        node2 = toNode(node2)
        super().__init__(node1.value + node2.value, [node1, node2])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad * np.ones_like(self.parents[0].grad), self.parents[0].grad.shape )
        self.parents[1].grad += _matchShape( self.grad * np.ones_like(self.parents[0].grad), self.parents[1].grad.shape )


class _Negation(Node):
    def __init__(self, node: Node|np.ndarray|float):
        node = toNode(node)
        super().__init__(-node.value, [node])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad * -np.ones_like(self.parents[0].grad), self.parents[0].grad.shape )


class _Product(Node):
    def __init__(self, node1: Node|np.ndarray|float, node2: Node|np.ndarray|float):
        node1 = toNode(node1)
        node2 = toNode(node2)
        super().__init__(node1.value * node2.value, [node1, node2])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad * self.parents[1].value, self.parents[0].grad.shape )
        self.parents[1].grad += _matchShape( self.grad * self.parents[0].value, self.parents[1].grad.shape )


class _Division(Node):
    def __init__(self, node1: Node|np.ndarray|float, node2: Node|np.ndarray|float):
        node1 = toNode(node1)
        node2 = toNode(node2)
        super().__init__(node1.value / node2.value, [node1, node2])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad / self.parents[1].value, self.parents[0].grad.shape )
        self.parents[1].grad += _matchShape( self.grad * (-self.parents[0].value / (self.parents[1].value**2)), self.parents[1].grad.shape )


class _Power(Node):
    def __init__(self, base_node: Node|np.ndarray|float, exp_node: Node|np.ndarray|float):
        base_node = toNode(base_node)
        exp_node = toNode(exp_node)
        super().__init__(base_node.value ** exp_node.value, [base_node, exp_node])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad * self.parents[1].value * (self.parents[0].value**(self.parents[1].value-1)), self.parents[0].grad.shape)
        self.parents[1].grad += _matchShape( self.grad * (self.parents[0].value**self.parents[1].value) * np.log(self.parents[0].value), self.parents[1].grad.shape)


class _NaturalLog(Node):
    def __init__(self, node: Node|np.ndarray|float):
        node = toNode(node)
        super().__init__(np.log(node.value), [node])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad * (1/self.parents[0].value), self.parents[0].grad.shape )


class _Exponential(Node):
    def __init__(self, node: Node|np.ndarray|float):
        node = toNode(node)
        super().__init__(np.exp(node.value), [node])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad * np.exp(self.parents[0].value), self.parents[0].grad.shape )


class _MatrixMultiplication(Node):
    def __init__(self, node1: Node|np.ndarray|float, node2: Node|np.ndarray|float):
        node1 = toNode(node1)
        node2 = toNode(node2)
        super().__init__(node1.value @ node2.value, [node1, node2])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad @ self.parents[1].value.T, self.parents[0].grad.shape )
        self.parents[1].grad += _matchShape( self.parents[0].value.T @ self.grad, self.parents[1].grad.shape )


class _Transpose(Node):
    def __init__(self, node: Node|np.ndarray|float):
        node = toNode(node)
        super().__init__(node.value.T, [node])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad.T, self.parents[0].grad.shape )


class _Summation(Node):
    def __init__(self, node: Node|np.ndarray, axis: int|None=None, keepdims: bool=False):
        node = toNode(node)
        super().__init__(np.sum(node.value, axis=axis, keepdims=keepdims), [node])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad * np.ones_like(self.parents[0].grad), self.parents[0].grad.shape )


class _Max(Node):
    def __init__(self, node: Node|np.ndarray):
        node = toNode(node)
        super().__init__(np.max(node.value), [node])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad * (self.parents[0].value == self.value), self.parents[0].grad.shape )


class _Concatenate(Node):
    def __init__(self, nodes: list[Node]|list[np.ndarray], axis: int=-1):
        for i in range(len(nodes)): nodes[i] = toNode(nodes[i])
        super().__init__(np.concatenate([node.value for node in nodes], axis=axis), nodes)
        self.axis = axis

    def storeDerivatives(self):
        for i in range(len(self.parents)):
            unconcat_slice = [ slice(None, None) ]*len(self.grad.shape)
            unconcat_slice[self.axis] = slice(i, i+1)
            self.parents[i].grad += self.grad[*unconcat_slice]


class _Flatten(Node):
    def __init__(self, node: Node|np.ndarray):
        node = toNode(node)
        super().__init__(node.value.flatten(), [node])

    def storeDerivatives(self):
        self.parents[0].grad += self.grad.flatten().reshape(self.parents[0].grad.shape)


class _Reshape(Node):
    def __init__(self, node: Node|np.ndarray, new_shape):
        node = toNode(node)
        super().__init__(node.value.reshape(new_shape), [node])

    def storeDerivatives(self):
        self.parents[0].grad += self.grad.flatten().reshape(self.parents[0].grad.shape)


class _Pad(Node):
    def __init__(self, node: Node|np.ndarray, pad_width, mode, **kwargs):
        node = toNode(node)
        super().__init__(np.pad(node.value, pad_width, mode, **kwargs), [node])
        self.pad_width = [ p if len(p) == 2 else (p[0], p[0]) for p in pad_width ]

    def storeDerivatives(self):
        grad_slice = [ slice(p[0], -p[1]) if (p[0] != 0 and p[1] != 0) else slice(None, None) for p in self.pad_width ]
        self.parents[0].grad += self.grad[*grad_slice]


class _Slice(Node):
    def __init__(self, node: Node|np.ndarray, idx_slice: slice):
        node = toNode(node)
        super().__init__(node.value[idx_slice], [node])
        self.idx_slice = idx_slice

    def storeDerivatives(self):
        self.parents[0].grad += self.grad[self.idx_slice]


class ReLU(Node):
    def __init__(self, node: Node|np.ndarray|float):
        node = toNode(node)
        super().__init__(np.maximum(node.value, 0), [node])

    def storeDerivatives(self):
        self.parents[0].grad += _matchShape( self.grad * (self.value > 0).astype(float), self.parents[0].grad.shape )