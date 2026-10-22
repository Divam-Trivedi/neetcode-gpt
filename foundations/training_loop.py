import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        w = np.zeros(len(X[0]))
        b = 0
        for ep in range(epochs):
            y_hat = X @ w + b # Forward Pass
            loss = np.mean((y_hat-y)**2) # Loss Compute
            dL_w = (2/len(X)) * X.T @ (y_hat - y) # Gradient for Weight
            dL_b = 2*np.mean(y_hat - y) # Gradient for Bias
            w = w - lr*dL_w # Update Weight
            b = b - lr*dL_b # Update Bias
        return (np.round(w, 5), round(b, 5))
