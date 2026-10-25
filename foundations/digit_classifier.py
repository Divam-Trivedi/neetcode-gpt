import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.input_layer = torch.nn.Linear(784, 512)
        self.activation_layer = torch.nn.ReLU()
        self.dropout_layer = torch.nn.Dropout(p=0.2)
        self.output_layer = torch.nn.Linear(512, 10)

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        output = self.input_layer(images)
        output = self.activation_layer(output)
        output = self.dropout_layer(output)
        output = self.output_layer(output)
        output = torch.sigmoid(output)
        return torch.round(output, decimals=4)
