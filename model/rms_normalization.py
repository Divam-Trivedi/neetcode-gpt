import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        sum = 0
        for num in x:
            sum += num**2
        rms = np.sqrt(sum/len(x) + eps)
        x_hat = x / rms
        return list(np.round(gamma * x_hat, 4))
