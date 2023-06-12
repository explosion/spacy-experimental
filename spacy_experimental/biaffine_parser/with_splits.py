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
    [split_len, hidden_size]) and feeds them to the inner model.

    A quirk is that the layer outputs per split rather than per doc. The reason
    is that the arc predicter will output splits with the shape [split_len,
    split_len]. Since the splits within a doc have different lengths, we cannot
    concatenate them into a single doc representation (unless we add padding).
    This is not an issue, since the arc predicter knows how to deal with
    this output.
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
        X_unflat = _docs_to_splits(X_flat, lengths)
        model.layers[0].initialize(X=X_unflat, Y=Y)


def with_splits_forward(
    model: Model[InT, OutT],
    X_lengths: InT,
    is_train: bool,
) -> Tuple[OutT, Callable[[OutT], InT]]:
    inner: Model[InnerInT, InnerOutT] = model.layers[0]

    X, lengths = X_lengths

    X_flat = _docs_to_splits(X, lengths)
    Y_unflat, backprop_inner = inner(X_flat, is_train)

    def backprop(dY: OutT) -> InT:
        dX_unflat = backprop_inner(dY)
        return _splits_to_docs(model.ops, dX_unflat, lengths), lengths

    return Y_unflat, backprop


def _docs_to_splits(
    X_docs: List[Floats2d], docs_split_lengths: List[Ints1d]
) -> List[Floats2d]:
    X_splits: List[Floats2d] = []
    for X_doc, doc_split_lengths in zip(X_docs, docs_split_lengths):
        split_offsets = lengths2offsets(doc_split_lengths)
        X_splits.extend(
            X_doc[split_offset : split_offset + split_length]
            for split_offset, split_length in zip(split_offsets, doc_split_lengths)
        )
    return X_splits


def _splits_to_docs(
    ops: Ops, X_splits: List[Floats2d], docs_split_lengths: List[Ints1d]
) -> List[Floats2d]:
    X_docs = []
    for doc_split_lengths in docs_split_lengths:
        inner = X_splits[: len(doc_split_lengths)]
        X_splits = X_splits[len(doc_split_lengths) :]
        X_docs.append(ops.flatten(inner))
    return X_docs
