import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        z1 = np.array(np.dot(W1, x) + b1)
        a1 = np.array(np.maximum(z1, 0))
        y_hat = np.array(np.dot(a1, np.array(W2).T) + b2)
        loss = 0
        for i,j in zip(y_hat, y_true):
            loss += (i - j)**2
        loss = loss/len(y_hat)

        # Network: x -> Linear1 -> ReLU -> Linear2 -> MSE loss
        dL2 = np.array(2*(y_hat-y_true) / len(y_hat)) # Gradient for the loss
        dW2 = np.array(np.outer(dL2.T, a1)) # Gradient for Linear2
        db2 = dL2 # Bias

        dL1 = dL2 @ W2 # Through output layer
        dL1 = dL1 * (z1>0) # Throiugh ReLU
        dW1 = np.array(np.outer(dL1, x))
        db1 = dL1
        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4),
            "db1": np.round(db1, 4),
            "dW2": np.round(dW2, 4),
            "db2": np.round(db2, 4)
        }


                
