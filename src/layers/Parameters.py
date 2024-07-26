from differentiation import Node


class Parameters:
    def __init__(self, node: Node):
        self.node = node


    def zero_grad(self):
        self.node.zero_grad()


    @property
    def grad(self):
        return self.node.grad


    @property
    def value(self):
        return self.node.value
    

    def step(self, step_size):
        self.node.value -= step_size 


    def __len__(self):
        return self.node.value.size