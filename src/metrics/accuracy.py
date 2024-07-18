import numpy as np



def categoricalAccuracy(preds: list[np.ndarray], labels: list[np.ndarray], average_over_batch=True) -> float:
    corrects = 0

    for pred, label in zip(preds, labels):
        if np.argmax(pred) == np.argmax(label):
            corrects += 1

    if average_over_batch:
        return corrects / len(preds)
    else:
        return corrects