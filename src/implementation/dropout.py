import torch

class Dropout(torch.nn.Module):
    def __init__(self, p: float = 0.5, mode: str = "inverted"):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("p must be in [0, 1)")
        self.p = p
        self.mode = mode

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return x
        
        # create a dropout mask (0 if dropped, 1 if kept)
        mask = (torch.rand_like(x) > self.p).float()
        
        if self.mode == "inverted":
            if self.training:
                # inverted dropout scaling. Scale at training
                return x * mask / (1.0 - self.p)
            else:
                return x
        
        elif self.mode == "standard":
            if self.training:
                return x * mask
            else:
                # standard dropout scaling. Scale at inference
                return x * (1.0 - self.p)

    