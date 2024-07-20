from tqdm.auto import tqdm
from layers import Layer
from losses import Loss
from optimizers import Optimizer
from data import DataLoader
from typing import Callable



class Trainer:
    def __init__(self, 
        model: Layer,
        loss: Loss,
        optimizer: Optimizer,
        train_data: DataLoader,
        val_data: DataLoader|None = None,
        metrics: list[(str, Callable)] = []
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_data = train_data
        self.val_data = val_data
        self.metrics = metrics
        self.tot_epochs = 0


    def train(self, epochs: int):
        history = []

        for epoch in (pbar := tqdm(range(epochs), position=0)):
            epoch_train_loss = 0
            epoch_train_metrics = [0] * len(self.metrics)
            epoch_val_loss = 0
            epoch_val_metrics = [0] * len(self.metrics)

            # Training
            for i, (inputs, labels) in enumerate(self.train_data):
                pbar.set_description(f"Batch {i+1}/{len(self.train_data)}")
                self.optimizer.zero_grad()
                preds = self.model(inputs)
                batch_loss = self.loss(preds, labels)
                batch_loss.backward()
                self.optimizer.step()

                epoch_train_loss += batch_loss.value * len(preds)
                for i, (_, metric_fn) in enumerate(self.metrics):
                    epoch_train_metrics[i] += metric_fn([p.value for p in preds], labels, average_over_batch=False)


            # Validation
            if self.val_data is not None:
                for i, (inputs, labels) in enumerate(self.val_data):
                    preds = self.model(inputs)
                    batch_loss = self.loss(preds, labels)

                    epoch_val_loss += batch_loss.value * len(preds)
                    for i, (_, metric_fn) in enumerate(self.metrics):
                        epoch_val_metrics[i] += metric_fn([p.value for p in preds], labels, average_over_batch=False)


            logs = ""
            epoch_train_loss = epoch_train_loss / self.train_data.size
            epoch_train_metrics = [ m / self.train_data.size for m in epoch_train_metrics ]
            train_metrics_str = "| ".join([f"train_{name}: {epoch_train_metrics[i]:.4f}" for i, (name, _) in enumerate(self.metrics)])
            logs += f"Epoch {self.tot_epochs} | train_loss: {epoch_train_loss:.4f} {train_metrics_str}"
            if self.val_data is not None:
                epoch_val_loss = epoch_val_loss / self.val_data.size
                epoch_val_metrics = [ m / self.val_data.size for m in epoch_val_metrics ]
                val_metrics_str = "| ".join([f"val_{name}: {epoch_val_metrics[i]:.4f}" for i, (name, _) in enumerate(self.metrics)])
                logs += f" || val_loss: {epoch_val_loss:.4f} {val_metrics_str}"
            self.tot_epochs += 1
            pbar.write(logs)

            history.append({
                "epoch": self.tot_epochs,
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "train_metrics": epoch_train_metrics,
                "val_metrics": epoch_val_metrics
            })

        return history