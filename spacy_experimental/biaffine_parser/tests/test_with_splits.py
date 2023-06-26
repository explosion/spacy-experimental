import numpy
from spacy_experimental.biaffine_parser.with_splits import with_splits

from .util import memoize


def test_with_splits():
    model = with_splits(memoize())
    doc1 = model.ops.xp.arange(9, dtype="float32").reshape(3, 3)
    doc2 = model.ops.xp.arange(6, dtype="float32").reshape(2, 3)
    _, backprop = model(
        ([doc1, doc2], [model.ops.asarray1i([1, 2]), model.ops.asarray1i([2])]), True
    )

    check_output = [
        model.ops.asarray2f([[0, 1, 2]]),
        model.ops.asarray2f([[3, 4, 5], [6, 7, 8]]),
        model.ops.asarray2f([[0, 1, 2], [3, 4, 5]]),
    ]

    X_inner = model.layers[0].attrs["X"]
    numpy.testing.assert_equal(
        X_inner,
        [check_output],
    )

    backprop(check_output)
    dY_inner = model.layers[0].attrs["dY"]
    numpy.testing.assert_equal(
        dY_inner,
        [check_output],
    )
