import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_z = max(z)
        total = np.sum([np.exp(i-max_z) for i in z])
        res = []
        for i in z:
            res.append(round(float(np.exp(i-max_z) / total), 4))
        return np.array(res)
