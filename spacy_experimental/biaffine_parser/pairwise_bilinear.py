from typing import Callable, List, Optional, Tuple, TypeVar, Union
from typing import cast
from spacy import registry
from spacy.tokens.doc import Doc
from thinc.api import Model, chain, get_width, torch2xp
from thinc.api import with_getitem, xp2torch
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.types import (
    ArgsKwargs,
    Floats2d,
    Floats3d,
    Floats4d,
    Ints1d,
)


from .with_minibatch_by_padded_size import with_minibatch_by_padded_size
from .with_pad_seq_unpad_matrix import with_pad_seq_unpad_matrix
from .with_splits import with_splits

# Ensure that the spacy-experimental package can register entry points without
# Torch installed.
PyTorchPairwiseBilinearModel: Optional[type]
try:
    from .pytorch_pairwise_bilinear import (
        PairwiseBilinearModel as PyTorchPairwiseBilinearModel,
    )
except ImportError:
    PyTorchPairwiseBilinearModel = None


# Model notes
#
# The Thinc part of the pairwire bilinear model used to be fairly simple: we
# would collect the splits from all documents and then pad them. However, this
# would run the parser out of GPU memory on very large docs. So instead, we do
# the following:
#
# - Get all splits and flatten to a list of split representations.
#   (with_splits)
# - Batch the splits by their padded sizes. This ensures that memory
#   use is constant when splits have a maximum size. This also permits
#   some buffering, so that we get more equisized batches.
#   (with_minibatch_by_padded_size)
# - The splits in the batches are padded and passed to the Torch model.
#   Since the outputs of the Torch model are matrices, we unpad taking
#   this into account. (with_pad_seq_unpad_matrix)
#
# In contrast to most with_* layers, with_splits is not symmetric. It
# takes at its input representations for each document (List[Floats2d]),
# however it outputs pairwise score matrices per split. The reason is that
# since the dimensions of the score matrices differ per split, we cannot
# concatenate them at a document level.


InT = TypeVar("InT")
OutT = TypeVar("OutT")


def build_pairwise_bilinear(
    tok2vec: Model[List[Doc], List[Floats2d]],
    nO=None,
    *,
    dropout: float = 0.1,
    hidden_width: int = 128,
    max_items: int = 4096,
    mixed_precision: bool = False,
    grad_scaler: Optional[PyTorchGradScaler] = None
) -> Model[Tuple[List[Doc], List[Ints1d]], List[Floats2d]]:
    if PyTorchPairwiseBilinearModel is None:
        raise ImportError(
            "PairwiseBiLinear layer requires PyTorch: pip install thinc[torch]"
        )

    nI = None
    if tok2vec.has_dim("nO") is True:
        nI = tok2vec.get_dim("nO")

    pairwise_bilinear: Model[Tuple[Floats3d, Ints1d], Floats3d] = Model(
        "pairwise_bilinear",
        forward=pairwise_bilinear_forward,
        init=pairwise_bilinear_init,
        dims={"nI": nI, "nO": nO},
        attrs={
            # We currently do not update dropout when dropout_rate is
            # changed, since we cannot access the underlying model.
            "dropout_rate": dropout,
            "hidden_width": hidden_width,
            "mixed_precision": mixed_precision,
            "grad_scaler": grad_scaler,
        },
    )

    model: Model[Tuple[List[Doc], List[Ints1d]], List[Floats2d]] = chain(
        cast(
            Model[Tuple[List[Doc], List[Ints1d]], Tuple[List[Floats2d], List[Ints1d]]],
            with_getitem(0, tok2vec),
        ),
        with_splits(
            with_minibatch_by_padded_size(
                with_pad_seq_unpad_matrix(pairwise_bilinear), size=max_items
            )
        ),
    )
    model.set_ref("pairwise_bilinear", pairwise_bilinear)

    return model


def pairwise_bilinear_init(model: Model, X=None, Y=None):
    if PyTorchPairwiseBilinearModel is None:
        raise ImportError(
            "PairwiseBiLinear layer requires PyTorch: pip install thinc[torch]"
        )

    if model.layers:
        return

    if X is not None and model.has_dim("nI") is None:
        model.set_dim("nI", get_width(X))
    if Y is not None and model.has_dim("nO") is None:
        model.set_dim("nO", get_width(Y))

    hidden_width = model.attrs["hidden_width"]
    mixed_precision = model.attrs["mixed_precision"]
    grad_scaler = model.attrs["grad_scaler"]

    PyTorchWrapper = registry.get("layers", "PyTorchWrapper.v2")
    model._layers = [
        PyTorchWrapper(
            PyTorchPairwiseBilinearModel(
                model.get_dim("nI"),
                model.get_dim("nO"),
                dropout=model.attrs["dropout_rate"],
                hidden_width=hidden_width,
            ),
            convert_inputs=convert_inputs,
            convert_outputs=convert_outputs,
            mixed_precision=mixed_precision,
            grad_scaler=grad_scaler,
        )
    ]


def pairwise_bilinear_forward(model: Model, X, is_train: bool):
    return model.layers[0](X, is_train)


def convert_inputs(
    model: Model, X_lengths: Tuple[Floats3d, Ints1d], is_train: bool = False
) -> Tuple[ArgsKwargs, Callable[[ArgsKwargs], Tuple[Floats3d, Ints1d]]]:
    """
    Shapes:
        X_lengths[0] - (n_splits, max_split_len, hidden_size)
        X_lengths[1] - (n_splits,)
    """
    X, L = X_lengths

    Xt = xp2torch(X, requires_grad=is_train)
    Lt = xp2torch(L)

    def convert_from_torch_backward(d_inputs: ArgsKwargs) -> Tuple[Floats3d, Ints1d]:
        dX = cast(Floats3d, torch2xp(d_inputs.args[0]))
        return dX, L

    output = ArgsKwargs(args=(Xt, Lt), kwargs={})

    return output, convert_from_torch_backward


def convert_outputs(
    model, inputs_outputs, is_train: bool
) -> Tuple[
    Union[Floats3d, Floats4d], Callable[[Union[Floats3d, Floats4d]], ArgsKwargs]
]:
    """
    Shapes:
        inputs_outputs[0][0] - (n_splits, max_split_len, hidden_size)
        inputs_outputs[0][1] - (n_splits,)
        inputs_outputs[1]    - (n_splits, max_split_len, max_split_len) or
                               (n_splits, max_split_len, max_split_len, n_class)
    """
    (_, lengths), Y_t = inputs_outputs

    def convert_for_torch_backward(dY: Union[Floats3d, Floats4d]) -> ArgsKwargs:
        dY_t = xp2torch(dY)
        return ArgsKwargs(
            args=([Y_t],),
            kwargs={"grad_tensors": [dY_t]},
        )

    Y = cast(Union[Floats3d, Floats4d], torch2xp(Y_t))

    return Y, convert_for_torch_backward
