import torch.nn as nn


class PFFlayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        feedforward_dim: int,
        dropout: float
    ) -> None:
        super(PFFlayer, self).__init__()

        self.pff = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, model_dim)
        )

    def forward(self, x):
        return self.pff(x)
