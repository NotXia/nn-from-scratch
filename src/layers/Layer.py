from abc import ABC, abstractmethod
from differentiation import Node



class Layer(ABC):
    @abstractmethod
    def __call__(self, batch: list[Node]):
        pass
    

    @abstractmethod
    def zero_grad(self):
        pass

    
    @abstractmethod
    def parameters(self):
        pass