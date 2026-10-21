import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists

        m = 0.1 # default momentum
        if training:
            mu_b = np.mean(x, axis=0) #mean across batch for each feature
            sigma_sq_b = np.mean((x - mu_b)**2, axis=0) #variance across batch
            x_hat = (x-mu_b) / np.sqrt(sigma_sq_b + eps)
            y = gamma * x_hat + beta
            running_mean = np.dot((1-m), running_mean) + m*mu_b
            running_var = np.dot((1-m), running_var) + m*sigma_sq_b
        else:
            y = (np.subtract(x, running_mean)) / np.sqrt(np.add(running_var, eps))
        
        return (np.round(y, 4), np.round(running_mean, 4), np.round(running_var, 4))