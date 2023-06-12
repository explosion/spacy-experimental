import pytest

pytest.importorskip("torch")

import torch

from torch.nn import Bilinear as TorchBilinear
from spacy_experimental.biaffine_parser.pytorch_bilinear import Bilinear


def test_bilinear_against_torch():
    # Disable biases, they are handled differently between our
    # implementation and Torch.
    torch_bilinear = TorchBilinear(5, 5, 7, bias=False)
    bilinear = Bilinear(5, 7, bias_u=False, bias_v=False)
    with torch.no_grad():
        bilinear.weight.copy_(torch_bilinear.weight)

    u = torch.rand((10, 5), dtype=torch.float32)
    v = torch.rand((10, 5), dtype=torch.float32)

    torch.testing.assert_close(bilinear(u, v), torch_bilinear(u, v))
