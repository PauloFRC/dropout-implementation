import torch

class Dropout(torch.nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("p must be in [0, 1)")
        self.p = p

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        
        # Create a dropout mask (0 if dropped, 1 if kept)
        mask = (torch.rand_like(x) > self.p).float()

        # inverted dropout scaling. TODO: FAZER TAMBÉM O PADRÃO O DO ARTIGO 
        return x * mask / (1.0 - self.p)
    