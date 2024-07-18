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
    

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_inputs = [ self.inputs[indexes[i]] for i in range(len(indexes)) ]
        batch_labels = [ self.labels[indexes[i]] for i in range(len(indexes)) ]
        return batch_inputs, batch_labels


    def __iter__(self):
        self.indexes = np.arange(len(self.inputs))
        if self.shuffle: 
            np.random.shuffle(self.indexes)
        self.curr_batch = 0
        return self


    def __next__(self):
        if self.curr_batch >= len(self): raise StopIteration

        batch_inputs, batch_labels = self[self.curr_batch]
        self.curr_batch += 1
        return batch_inputs, batch_labels