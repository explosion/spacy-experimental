import torch
from torch import nn


class BilinearModel(nn.Module):
    def __init__(
        self,
        nI: int,
        nO: int,
        *,
        activation: torch.nn.Module = nn.ReLU(),
        dropout: float = 0.1,
        hidden_width: int = 128,
    ):
        super(BilinearModel, self).__init__()

        self.head = nn.Linear(nI, hidden_width)
        self.dependent = nn.Linear(nI, hidden_width)
        self.bilinear = nn.Bilinear(hidden_width, hidden_width, nO)
        self.activation = activation
        self._dropout = nn.Dropout(dropout)

        # Default BiLinear initialization creates parameters that are
        # much too large, resulting in large regression.
        torch.nn.init.xavier_uniform_(self.bilinear.weight)

    @property
    def dropout(self) -> float:
        return self._dropout.p

    @dropout.setter
    def dropout(self, p: float):
        self._dropout.p = p

    def forward(self, x: torch.Tensor, heads: torch.Tensor):
        # Create representations of tokens as heads and dependents.
        head = self._dropout(self.activation(self.head(x[heads.long()])))
        dependent = self._dropout(self.activation(self.dependent(x)))

        logits = self.bilinear(head, dependent)

        if self.training:
            return logits.softmax(-1)
        else:
            return logits
