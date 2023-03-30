from typing import Callable, List, Optional, Tuple
from thinc.api import Model, Ops
from thinc.types import Floats2d, Ints1d

from ._util import lengths2offsets

InT = Tuple[List[Floats2d], List[Ints1d]]
OutT = List[Floats2d]
InnerInT = List[Floats2d]
InnerOutT = List[Floats2d]


def with_splits(inner: Model[InnerInT, InnerOutT]) -> Model[InT, OutT]:
    """This layer splits document representations (List[Floats2d], shape:
    [doc_len, hidden_size]) into split representations (List[Floats2d], shape:
    [n_splits, hidden_size]) and feeds them to the inner model.

    A quirk is that the layer outputs per split rather than per doc. The reason
    is that the arc predicter will output splits with the shape [split_len,
    split_len]. Since the splits within a doc have different lengths, we cannot
    concatenate then into a single doc representation (unless we add padding).
    This is not an issue, since the arc predictes knows how to deal with
    these output.
    """
    return Model(
        "splits",
        with_splits_forward,
        init=with_splits_init,
        layers=[inner],
    )


def with_splits_init(model: Model[InT, OutT], X: Optional[InT] = None, Y=None) -> None:
    if X is None:
        model.layers[0].initialize(X=X, Y=Y)
    else:
        X_flat, lengths = X
        X_unflat = _unflatten_inner_flatten_outer(X_flat, lengths)
        model.layers[0].initialize(X=X_unflat, Y=Y)


def with_splits_forward(
    model: Model[InT, OutT],
    X_lengths: InT,
    is_train: bool,
) -> Tuple[OutT, Callable[[OutT], InT]]:
    inner: Model[InnerInT, InnerOutT] = model.layers[0]

    X, lengths = X_lengths

    X_flat = _unflatten_inner_flatten_outer(X, lengths)
    Y_unflat, backprop_inner = inner(X_flat, is_train)

    def backprop(dY: OutT) -> InT:
        dX_unflat = backprop_inner(dY)
        return _unflatten_outer_flatten_inner(model.ops, dX_unflat, lengths), lengths

    return Y_unflat, backprop


def _unflatten_inner_flatten_outer(
    X: List[Floats2d], lengths: List[Ints1d]
) -> List[Floats2d]:
    X_flat: List[Floats2d] = []
    for X_inner, lengths_inner in zip(X, lengths):
        split_offsets = lengths2offsets(lengths_inner)
        X_flat.extend(
            X_inner[split_offset : split_offset + split_length]
            for split_offset, split_length in zip(split_offsets, lengths_inner)
        )
    return X_flat


def _unflatten_outer_flatten_inner(
    ops: Ops, X: List[Floats2d], lengths: List[Ints1d]
) -> List[Floats2d]:
    r = []
    for lengths_inner in lengths:
        inner = X[: len(lengths_inner)]
        X = X[len(lengths_inner) :]
        r.append(ops.flatten(inner))
    return r
