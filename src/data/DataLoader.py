import numpy as np
import random



class DataLoader:
    def __init__(self, inputs: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool=True):
        self.inputs = inputs
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle


    def __len__(self):
        return len(self.inputs) // self.batch_size
    

    def __iter__(self):
        self.indexes = [ *range(len(self.inputs)) ]
        if self.shuffle: 
            random.shuffle(self.indexes)
        return self


    def __next__(self):
        if len(self.indexes) < self.batch_size: raise StopIteration

        batch = []
        for _ in range(self.batch_size):
            choice = random.randint(0, len(self.indexes)-1)
            idx = self.indexes[choice]
            batch.append( (self.inputs[idx], self.labels[idx]) )
            self.indexes.pop(choice)
        return batch