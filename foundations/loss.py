import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        loss = 0;
        for i,j in zip(y_true,y_pred):
            j = np.clip(j, 1e-7, 1-1e-7)
            loss += i*np.log(j)+ (1-i)*np.log(1-j)
        return round(-loss/len(y_true), 4)
        

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        loss = 0
        for i,j in zip(y_true, y_pred):
            loss += np.sum(i*np.log(np.clip(j,1e-7, 1 - 1e-7)))
        return round(-loss/len(y_true), 4)
