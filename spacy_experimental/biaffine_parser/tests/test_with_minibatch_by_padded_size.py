from thinc.api import noop

from spacy_experimental.biaffine_parser.with_minibatch_by_padded_size import with_minibatch_by_padded_size

from .util import memoize


def test_batching():
    model = with_minibatch_by_padded_size(memoize(), size=18)
    X = ["peach", "banana", "apple", "pineapple"]
    _, backprop = model(X, True)
    backprop(["a", "b", "c", "d"])
    assert model.layers[0].attrs["X"] == [["pineapple"], ["peach", "banana", "apple"]]
    assert model.layers[0].attrs["dY"] == [["d"], ["a", "b", "c"]]


def test_output():
    # Check that outupts are in the same order as inputs.
    model = with_minibatch_by_padded_size(noop(), size=18)
    X = ["peach", "banana", "apple", "pineapple"]
    Y, backprop = model(X, True)
    assert Y == ["peach", "banana", "apple", "pineapple"]
    dX = backprop(["a", "b", "c", "d"])
    assert dX == ["a", "b", "c", "d"]
