import torch
from torch import nn
from torch.nn import functional as F


class BilinearModel(nn.Module):
    def __init__(
        self,
        nI: int,
        nO: int,
        *,
        activation=nn.GELU(),
        hidden_width=128,
    ):
        super(BilinearModel, self).__init__()

        self.head = nn.Linear(nI, hidden_width)
        self.dependent = nn.Linear(nI, hidden_width)
        self.bilinear = nn.Bilinear(hidden_width, hidden_width, nO)
        self.activation = activation
        self.dropout = nn.Dropout(0.1)

        # Default BiLinear initialization creates parameters that are
        # much too large, resulting in large regression.
        torch.nn.init.xavier_uniform_(self.bilinear.weight)

    def forward(self, x: torch.Tensor, heads: torch.Tensor):
        # Create representations of tokens as heads and dependents.
        head = self.dropout(self.activation(self.head(x[heads.long()])))
        dependent = self.dropout(self.activation(self.dependent(x)))

        logits = self.bilinear(head, dependent)

        if self.training:
            return logits.softmax(-1)
        else:
            return logits
