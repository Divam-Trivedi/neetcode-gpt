import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        res = []
        for x in X:
            res.append(np.round(np.sum(x*weights), 5))
        return np.array(res)
        

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        mse_loss = 0;
        for i, j in zip(model_prediction, ground_truth):
            mse_loss += (i-j)**2
        return np.round(mse_loss/len(model_prediction), 5)[0]
