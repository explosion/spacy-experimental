from typing import Callable, Tuple
import numpy
from thinc.api import Model

from spacy_experimental.biaffine_parser.with_pad_seq_unpad_matrix import (
    InnerInT,
    InnerOutT,
    with_pad_seq_unpad_matrix,
)


def _mock_model():
    return Model("mock_model", _mock_model_forward)


def _mock_model_forward(
    model: Model[InnerInT, InnerOutT], X: InnerInT, is_train: bool
) -> Tuple[InnerOutT, Callable[[InnerOutT], InnerInT]]:
    Xf, lens = X

    # Do something resembling our pairwise bilinear layer.
    Y = Xf @ Xf.transpose(0, 2, 1)  # type: ignore

    def backprop(dY: InnerOutT) -> InnerInT:
        model.attrs["last_dY"] = dY
        return Xf + 1.0, lens

    return Y, backprop


def test_with_pad_seq_unpad_matrix():
    model = with_pad_seq_unpad_matrix(_mock_model())
    X = [
        model.ops.xp.arange(6, dtype="float32").reshape(3, 2),
        model.ops.xp.arange(4, dtype="float32").reshape(2, 2),
    ]
    Y, backprop = model(
        X,
        True,
    )

    numpy.testing.assert_equal(
        Y,
        [
            model.ops.asarray2f([[1, 3, 5], [3, 13, 23], [5, 23, 41]], dtype="float32"),
            model.ops.asarray2f([[1, 3], [3, 13]]),
        ],
    )

    dY = [
        model.ops.xp.arange(9, dtype="float32").reshape(3, 3),
        model.ops.xp.arange(4, dtype="float32").reshape(2, 2),
    ]

    dX = backprop(dY)

    numpy.testing.assert_equal(dX, [X[0] + 1.0, X[1] + 1.0])
    numpy.testing.assert_equal(
        model.layers[0].attrs["last_dY"],
        [dY[0], model.ops.asarray2f([[0, 1, 0], [2, 3, 0], [0, 0, 0]])],
    )
