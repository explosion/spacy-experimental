import torch
from torch import nn
from torch.nn import functional as F


class VariationalDropout(nn.Module):
    """Variational dropout (Gal and Ghahramani, 2016)"""

    def __init__(self, p: float):
        super(VariationalDropout, self).__init__()
        self.p = p

    def forward(self, x: torch.Tensor):
        if not self.training:
            return x

        batch_size, _, repr_size = x.shape
        dropout_mask = F.dropout(
            torch.ones((batch_size, 1, repr_size), device=x.device), self.p, self.training
        )

        return x * dropout_mask


class PairwiseBilinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, *, bias_u=True, bias_v=True
    ):
        super(PairwiseBilinear, self).__init__()

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
        assert u.shape == v.shape, "Inputs to PairwiseBilinear must have the same shape"
        assert len(u.shape) == 3, "Inputs to PairwiseBilinear must have a 3d shape"

        batch_size, seq_len, _ = u.shape

        ones = torch.ones((batch_size, seq_len, 1), dtype=u.dtype, device=u.device)

        if self.bias_u:
            u = torch.cat([u, ones], -1)

        if self.bias_v:
            v = torch.cat([v, ones], -1)

        # Ideally we'd want to compute:
        #
        # torch.einsum("blu,ouv,bmv->bmlo", u, self.weight, v)
        #
        # Although this works correctly for prediction, this seems to
        # lead to extreme gradients. Maybe this is an upstream bug?
        # So, we will do this in two steps.

        intermediate = torch.einsum("blu,ouv->blov", u, self.weight)
        return torch.einsum("bmv,blov->bmlo", v, intermediate)


class PairwiseBilinearModel(nn.Module):
    def __init__(
        self,
        nI: int,
        nO: int,
        *,
        activation: torch.nn.Module = nn.ReLU(),
        dropout: float = 0.1,
        hidden_width: int = 128,
    ):
        super(PairwiseBilinearModel, self).__init__()

        self.head = nn.Linear(nI, hidden_width)
        self.dependent = nn.Linear(nI, hidden_width)
        self.bilinear = PairwiseBilinear(hidden_width, nO)
        self.activation = activation
        self._dropout = VariationalDropout(dropout)

    @property
    def dropout(self) -> float:
        return self._dropout.p

    @dropout.setter
    def dropout(self, p: float):
        self._dropout.p = p

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor):
        max_seq_len = x.shape[1]

        token_mask = torch.arange(max_seq_len, device=x.device).unsqueeze(0) < seq_lens.unsqueeze(1)
        logits_mask = (token_mask.float() - 1.0) * 10000.0
        logits_mask = logits_mask.unsqueeze(1).unsqueeze(-1)

        # Create representations of tokens as heads and dependents.
        head = self._dropout(self.activation(self.head(x)))
        dependent = self._dropout(self.activation(self.dependent(x)))

        # Compute biaffine attention matrix. This computes from the hidden
        # representations of the shape [batch_size, seq_len, hidden_width] the
        # attention matrices [batch_size, seq_len, seq_len, n_O].
        logits = self.bilinear(head, dependent)

        # Mask out head candidates that are padding time steps. The logits mask
        # has shape [batch_size, seq_len], we reshape it to [batch_size, 1,
        # seq_len] to mask out the head predictions.
        logits += logits_mask

        # If there is only one output feature, remove the last dimension.
        logits = logits.squeeze(-1)

        if self.training:
            return logits.softmax(-1)
        else:
            return logits
