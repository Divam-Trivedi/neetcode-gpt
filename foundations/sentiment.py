import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.embedding = torch.nn.Embedding(vocabulary_size, 16)
        self.linear = torch.nn.Linear(16, 1)

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        embedding = self.embedding(x)
        average = torch.mean(embedding, dim = 1)
        linear_output = self.linear(average)
        result = torch.sigmoid(linear_output)
        return torch.round(result, decimals=4)
        
