import torch
from torch import nn


class Bilinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, *, bias_u=True, bias_v=True
    ):
        super(Bilinear, self).__init__()

        self.bias_u = bias_u
        self.bias_v = bias_v

        bias_u_dim = 1 if bias_u else 0
        bias_v_dim = 1 if bias_v else 0

        self.weight = nn.parameter.Parameter(
            torch.empty(
                out_features, in_features + bias_u_dim, in_features + bias_v_dim
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, u: torch.Tensor, v: torch.Tensor):
        assert u.shape == v.shape, "Inputs to Bilinear must have the same shape"
        assert len(u.shape) == 2, "Inputs to Bilinear must have a 2d shape"

        batch_size, _ = u.shape

        ones = torch.ones((batch_size, 1), dtype=u.dtype, device=u.device)

        if self.bias_u:
            u = torch.cat([u, ones], -1)

        if self.bias_v:
            v = torch.cat([v, ones], -1)

        return torch.einsum("bu,bv,ouv->bo", u, v, self.weight)


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
        self.bilinear = Bilinear(hidden_width, nO)
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
