from typing import Callable, List, Optional, Tuple, cast
from thinc.api import Model, Ops
from thinc.types import Floats2d, Floats3d, Ints1d

InT = List[Floats2d]
OutT = List[Floats2d]
InnerInT = Tuple[Floats3d, Ints1d]
InnerOutT = Floats3d


def with_pad_seq_unpad_matrix(inner: Model[InnerInT, InnerOutT]) -> Model[InT, OutT]:
    """This layer is similar to with_padded, however it unpads
    correctly for layers that take sequences and output matrices."""
    return Model(
        "with_pad_seq_unpad_bilinear",
        with_pad_seq_unpad_matrix_forward,
        init=with_pad_seq_unpad_matrix_init,
        layers=[inner],
    )


def with_pad_seq_unpad_matrix_init(
    model: Model[InT, OutT], X: Optional[InT] = None, Y=None
):
    inner = model.layers[0]
    if X is not None:
        lengths = [len(Xs) for Xs in X]
        inner.initialize((model.ops.pad(X), model.ops.asarray1i(lengths)), Y)
    else:
        inner.initialize(X, Y)


def with_pad_seq_unpad_matrix_forward(
    model: Model[InT, OutT],
    X: InT,
    is_train: bool,
) -> Tuple[OutT, Callable[[OutT], InT]]:
    inner = model.layers[0]

    lengths = [len(Xs) for Xs in X]

    X_padded = model.ops.pad(X)
    Y_padded, backprop_inner = inner((X_padded, model.ops.asarray1i(lengths)), is_train)
    Y = unpad_matrix(Y_padded, lengths)

    def backprop(dY: List[Floats2d]) -> List[Floats2d]:
        dY_padded = pad_matrix(model.ops, dY)
        dX_padded, _ = backprop_inner(dY_padded)
        dX = model.ops.unpad(dX_padded, lengths)
        return cast(List[Floats2d], dX)

    return Y, backprop


def pad_matrix(ops: Ops, seqs: List[Floats2d], round_to=1) -> Floats3d:
    """Perform padding on a list of arrays so that they each have the same
    length, by taking the maximum dimension across each axis. This only
    works on non-empty sequences with the same `ndim` and `dtype`.

    Different from Thinc, because it operates on a matrix with padding
    in two dimensions.
    """
    if not seqs:
        raise ValueError("Cannot pad empty sequence")
    if len(set(seq.ndim for seq in seqs)) != 1:
        raise ValueError("Cannot pad sequences with different ndims")
    if len(set(seq.dtype for seq in seqs)) != 1:
        raise ValueError("Cannot pad sequences with different dtypes")
    if any(len(seq.shape) != 2 or seq.shape[0] != seq.shape[1] for seq in seqs):
        raise ValueError("Cannot pad non-square matrices")
    # Find the maximum dimension along each axis. That's what we'll pad to.
    length = max(seq.shape[0] for seq in seqs)
    # Round the length to nearest bucket -- helps on GPU, to make similar
    # array sizes.
    length = (length + (round_to - 1)) // round_to * round_to
    final_shape = (len(seqs), length, length)
    output = ops.alloc3f(*final_shape)
    for i, arr in enumerate(seqs):
        # It's difficult to convince this that the dtypes will match.
        output[i, : arr.shape[0], : arr.shape[1]] = arr  # type: ignore[assignment, call-overload]
    return output


def unpad_matrix(padded: Floats3d, lengths: List[int]) -> List[Floats2d]:
    """The reverse/backward operation of the `pad` function: transform an
    array back into a list of arrays, each with their original length.

    Different from Thinc, because it operates on a matrix with padding
    in two dimensions.
    """
    output = []
    for i, length in enumerate(lengths):
        output.append(padded[i, :length, :length])
    return cast(List[Floats2d], output)
