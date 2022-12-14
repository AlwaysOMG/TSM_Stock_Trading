# import packages
import numpy as np

# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(FNN,self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        y = self.l3(x)
        return y

class DQN:
    def __init__(self) -> None:
        pass