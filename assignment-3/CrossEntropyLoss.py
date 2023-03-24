import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.eps = 1e-15  # avoid taking log of zero

    def forward(self, y_pred, y_true):
        m = y_true.shape[0]
        self.y_pred = y_pred
        self.y_true = y_true
        loss = -np.sum(y_true * np.log(y_pred + self.eps) +
                       (1 - y_true) * np.log(1 - y_pred + self.eps)) / m
        return loss

    def backward(self):
        m = self.y_true.shape[0]
        d_loss = (self.y_pred - self.y_true) / \
            (self.y_pred * (1 - self.y_pred) + self.eps) / m

        return d_loss
