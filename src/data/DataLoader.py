import numpy as np
import random



class DataLoader:
    def __init__(self, inputs: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool=True):
        assert len(inputs) == len(labels)
        self.inputs = np.expand_dims(np.array(inputs), axis=-1) # Extra dimension for easier dot product
        self.labels = np.array(labels)
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

        batch_inputs = []
        batch_labels = []
        for _ in range(self.batch_size):
            choice = random.randint(0, len(self.indexes)-1)
            idx = self.indexes[choice]
            batch_inputs.append( self.inputs[idx] )
            batch_labels.append( self.labels[idx] )
            self.indexes.pop(choice)
        return batch_inputs, batch_labels