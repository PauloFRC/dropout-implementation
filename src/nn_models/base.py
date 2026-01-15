import torch.nn as nn
from abc import ABC, abstractmethod

class BaseDropoutModel(nn.Module, ABC):
    def __init__(self, name, dropout_rate=0.4, dropout_mode="inverted"):
        super().__init__()
        self.name = name
        self.dropout_rate = dropout_rate
        self.dropout_mode = dropout_mode
    
    @abstractmethod
    def forward(self, x):
        pass