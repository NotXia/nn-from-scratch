from tqdm.auto import tqdm
from layers import Layer
from losses import Loss
from optimizers import Optimizer
from data import DataLoader
from differentiation import Node



class Trainer:
    def __init__(self, 
        model: Layer,
        loss: Loss,
        optimizer: Optimizer,
        train_data: DataLoader,
        val_data: DataLoader|None = None
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_data = train_data
        self.val_data = val_data
        self.tot_epochs = 0


    def train(self, epochs: int):
        for epoch in range(epochs):
            epoch_train_loss = 0

            for inputs, labels in self.train_data:
                self.optimizer.zero_grad()
                preds = self.model(inputs)
                batch_loss = self.loss(preds, labels)
                batch_loss.backward()
                self.optimizer.step()

                epoch_train_loss += batch_loss.value

            epoch_train_loss = epoch_train_loss / len(self.train_data)
            print(f"Epoch {epoch+1} | train_loss: {epoch_train_loss:.4f}")
            self.tot_epochs += 1
        
        return self.model