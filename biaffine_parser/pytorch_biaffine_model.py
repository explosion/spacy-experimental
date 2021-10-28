import math
import torch
from torch import nn
from torch.nn import functional as F


class VariationalDropout(nn.Module):
    """Variational dropout (Gal and Ghahramani, 2016)"""

    def __init__(self, p: float = 0.1):
        super(VariationalDropout, self).__init__()
        self.p = p

    def forward(self, x: torch.Tensor):
        if not self.training:
            return x

        batch_size, _, repr_size = x.shape
        dropout_mask = F.dropout(
            torch.ones((batch_size, 1, repr_size)), self.p, self.training
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

        ones = torch.ones((batch_size, seq_len, 1), dtype=u.dtype)

        if self.bias_u:
            u = torch.cat([u, ones], -1)

        if self.bias_v:
            v = torch.cat([v, ones], -1)

        # [batch_size, seq_len, out_features, v features].
        intermediate = torch.einsum("blu,ouv->blov", u, self.weight)

        # Perform a matrix multiplication to get the output with
        # the shape [batch_size, seq_len, seq_len, out_features].
        return torch.einsum("bmv,blov->bmlo", v, intermediate)

        # return torch.einsum("blu,ouv,bmv->bmlo", u, self.weight, v)


class BiaffineModel(nn.Module):
    def __init__(
        self,
        nI: int,
        nO: int,
        *,
        activation=nn.GELU(),
        arc_hidden_size=128,
        dep_hidden_size=256
    ):
        super(BiaffineModel, self).__init__()

        self.activation = activation

        self.head_arc = nn.Linear(nI, arc_hidden_size)
        self.dependent_arc = nn.Linear(nI, arc_hidden_size)
        self.bilinear_arc = PairwiseBilinear(arc_hidden_size, 1)

        self.head_label = nn.Linear(nI, dep_hidden_size)
        self.dependent_label = nn.Linear(nI, dep_hidden_size)
        self.bilinear_label = PairwiseBilinear(dep_hidden_size, nO)

        self.dropout = VariationalDropout()

    def forward(self, x, seq_lens):
        max_seq_len = x.shape[1]

        token_mask = torch.arange(max_seq_len).unsqueeze(0) < seq_lens.unsqueeze(1)
        logits_mask = (token_mask.float() - 1.0) * 10000.0
        logits_mask = logits_mask.unsqueeze(1)

        # Create representations of tokens as heads and dependents.
        head_arc = self.dropout(self.activation(self.head_arc(x)))
        dependent_arc = self.dropout(self.activation(self.dependent_arc(x)))
        head_label = self.dropout(self.activation(self.head_label(x)))
        dependent_label = self.dropout(self.activation(self.dependent_label(x)))

        # Compute biaffine attention matrix. This computes from the hidden
        # representations of the shape [batch_size, seq_len, hidden_size] the
        # attention matrices [batch_size, seq_len, seq_len].
        logits_arc = self.bilinear_arc(head_arc, dependent_arc).squeeze(-1)

        logits_label = self.bilinear_label(head_label, dependent_label)

        # Mask out head candidates that are padding time steps. The logits mask
        # has shape [batch_size, seq_len], we reshape it to [batch_size, 1,
        # seq_len] to mask out the head predictions.
        logits_arc += logits_mask
        logits_label += logits_mask.unsqueeze(-1)

        if self.training:
            # Compute head probability distribution.
            logits_arc = logits_arc.softmax(-1)
            logits_label = logits_label.softmax(-1)

        return logits_arc, logits_label
