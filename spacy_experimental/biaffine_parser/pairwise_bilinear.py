from typing import List, Optional, Tuple, cast

from spacy import registry
from spacy.tokens.doc import Doc
from thinc.api import Model, chain, get_width, list2array, torch2xp
from thinc.api import with_getitem, xp2torch
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.types import ArgsKwargs, Floats2d, Floats3d, Floats4d, Ints1d

# Ensure that the spacy-experimental package can register entry points without
# Torch installed.
try:
    from .pytorch_pairwise_bilinear import (
        PairwiseBilinearModel as PyTorchPairwiseBilinearModel,
    )
except ImportError:
    PyTorchPairwiseBilinearModel = None


def build_pairwise_bilinear(
    tok2vec: Model[List[Doc], List[Floats2d]],
    nO=None,
    *,
    dropout: float = 0.1,
    hidden_width: int = 128,
    mixed_precision: bool = False,
    grad_scaler: Optional[PyTorchGradScaler] = None
) -> Model[Tuple[List[Doc], Ints1d], Floats2d]:
    if PyTorchPairwiseBilinearModel is None:
        raise ImportError(
            "PairwiseBiLinear layer requires PyTorch: pip install thinc[torch]"
        )

    nI = None
    if tok2vec.has_dim("nO") is True:
        nI = tok2vec.get_dim("nO")

    pairwise_bilinear: Model[Tuple[Floats2d, Ints1d], Floats2d] = Model(
        "pairwise_bilinear",
        forward=pairswise_bilinear_forward,
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

    model = chain(
        cast(
            Model[Tuple[List[Doc], Ints1d], Tuple[Floats2d, Ints1d]],
            with_getitem(
                0, chain(tok2vec, cast(Model[List[Floats2d], Floats2d], list2array()))
            ),
        ),
        pairwise_bilinear,
    )
    model.set_ref("pairwise_bilinear", pairwise_bilinear)

    return model


def pairwise_bilinear_init(model: Model, X=None, Y=None):
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


def pairswise_bilinear_forward(model: Model, X, is_train: bool):
    return model.layers[0](X, is_train)


def convert_inputs(
    model: Model, X_lenghts: Tuple[Floats2d, Ints1d], is_train: bool = False
):
    flatten = model.ops.flatten
    unflatten = model.ops.unflatten
    pad = model.ops.pad
    unpad = model.ops.unpad

    X, L = X_lenghts

    Xt = xp2torch(pad(unflatten(X, L)), requires_grad=is_train)
    Lt = xp2torch(L)

    def convert_from_torch_backward(d_inputs: ArgsKwargs) -> Tuple[Floats2d, Ints1d]:
        dX = cast(Floats3d, torch2xp(d_inputs.args[0]))
        return cast(Floats2d, flatten(unpad(dX, list(L)))), L

    output = ArgsKwargs(args=(Xt, Lt), kwargs={})

    return output, convert_from_torch_backward


def convert_outputs(model, inputs_outputs, is_train):
    flatten = model.ops.flatten
    unflatten = model.ops.unflatten
    pad = model.ops.pad
    unpad = model.ops.unpad

    (_, lengths), Y_t = inputs_outputs

    def convert_for_torch_backward(dY: Tuple[Floats2d, Floats3d]) -> ArgsKwargs:
        dY_t = xp2torch(pad(unflatten(dY, lengths)))
        return ArgsKwargs(
            args=([Y_t],),
            kwargs={"grad_tensors": [dY_t]},
        )

    Y = cast(Floats4d, torch2xp(Y_t))
    Y = flatten(unpad(Y, lengths))

    return Y, convert_for_torch_backward
