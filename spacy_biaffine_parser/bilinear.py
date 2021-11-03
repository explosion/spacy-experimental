from spacy import registry
from spacy.ml import extract_spans
from spacy.tokens.doc import Doc
from thinc.layers.pytorchwrapper import PyTorchWrapper_v2
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
import torch.nn as nn
from thinc.api import (
    Model,
    Padded,
    PyTorchWrapper,
    Ragged,
    chain,
    get_width,
    list2array,
    list2ragged,
    with_array,
    with_getitem,
    with_padded,
    xp2torch,
    torch2xp,
    tuplify,
)
from thinc.types import ArgsKwargs, Array2d, Floats2d, Floats3d, Floats4d, Ints1d
from typing import Any, Optional, List, Tuple, cast

from .pytorch_bilinear import BilinearModel as PyTorchBilinearModel


@registry.architectures("Bilinear.v1")
def build_bilinear(
    tok2vec: Model[List[Doc], List[Floats2d]],
    nO=None,
    *,
    hidden_width: int = 128,
    mixed_precision: bool = False,
    grad_scaler: Optional[PyTorchGradScaler] = None
):
    nI = None
    if tok2vec.has_dim("nO") is True:
        nI = tok2vec.get_dim("nO")

    bilinear = Model(
        "bilinear",
        forward=bilinear_forward,
        init=bilinear_init,
        dims={"nI": nI, "nO": nO},
        attrs={
            "hidden_width": hidden_width,
            "mixed_precision": mixed_precision,
            "grad_scaler": grad_scaler,
        },
    )

    model = chain(
        with_getitem(0, chain(tok2vec, list2array())),
        bilinear,
    )
    model.set_ref("bilinear", bilinear)

    return model


def bilinear_init(model: Model, X=None, Y=None):
    if model.layers:
        return

    if X is not None and model.has_dim("nI") is None:
        model.set_dim("nI", get_width(X))
    if Y is not None and model.has_dim("nO") is None:
        model.set_dim("nO", get_width(Y))

    hidden_width = model.attrs["hidden_width"]
    mixed_precision = model.attrs["mixed_precision"]
    grad_scaler = model.attrs["grad_scaler"]

    model._layers = [
        PyTorchWrapper_v2(
            PyTorchBilinearModel(
                model.get_dim("nI"),
                model.get_dim("nO"),
                hidden_width=hidden_width,
            ),
            convert_inputs=convert_inputs,
            convert_outputs=convert_outputs,
            mixed_precision=mixed_precision,
            grad_scaler=grad_scaler,
        )
    ]


def bilinear_forward(model: Model, X, is_train: bool):
    return model.layers[0](X, is_train)


def convert_inputs(
    model: Model, X_heads: Tuple[Ragged, Ints1d], is_train: bool = False
):
    flatten = model.ops.flatten
    unflatten = model.ops.unflatten
    pad = model.ops.pad
    unpad = model.ops.unpad

    Xr, heads = X_heads

    Xt = xp2torch(Xr, requires_grad=is_train)
    Ht = xp2torch(heads)

    def convert_from_torch_backward(d_inputs: ArgsKwargs) -> Tuple[Floats2d, Ints1d]:
        dX = cast(Floats2d, torch2xp(d_inputs.args[0]))
        return dX, heads

    output = ArgsKwargs(args=(Xt, Ht), kwargs={})

    return output, convert_from_torch_backward


def convert_outputs(model, inputs_outputs, is_train):
    flatten = model.ops.flatten
    unflatten = model.ops.unflatten
    pad = model.ops.pad
    unpad = model.ops.unpad

    _, Y_t = inputs_outputs

    def convert_for_torch_backward(dY: Tuple[Floats2d, Floats3d]) -> ArgsKwargs:
        dY_t = xp2torch(dY)
        return ArgsKwargs(
            args=([Y_t],),
            kwargs={"grad_tensors": [dY_t]},
        )

    Y = cast(Floats2d, torch2xp(Y_t))

    return Y, convert_for_torch_backward
