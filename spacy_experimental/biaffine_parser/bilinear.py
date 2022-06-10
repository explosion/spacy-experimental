from typing import List, Optional, Tuple, cast

from spacy import registry
from spacy.tokens.doc import Doc
from thinc.api import Model, chain, get_width, list2array, torch2xp
from thinc.api import with_getitem, xp2torch
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.types import ArgsKwargs, Floats2d, Ints1d


# Ensure that the spacy-experimental package can register entry points without
# Torch installed.
try:
    from .pytorch_bilinear import BilinearModel as PyTorchBilinearModel
except ImportError:
    PyTorchBilinearModel = None


def build_bilinear(
    tok2vec: Model[List[Doc], List[Floats2d]],
    nO: Optional[int] = None,
    *,
    dropout: float = 0.1,
    hidden_width: int = 128,
    mixed_precision: bool = False,
    grad_scaler: Optional[PyTorchGradScaler] = None
) -> Model[Tuple[List[Doc], Ints1d], Floats2d]:
    if PyTorchBilinearModel is None:
        raise ImportError("BiLinear layer requires PyTorch: pip install thinc[torch]")

    nI = None
    if tok2vec.has_dim("nO") is True:
        nI = tok2vec.get_dim("nO")

    bilinear: Model[Tuple[Floats2d, Ints1d], Floats2d] = Model(
        "bilinear",
        forward=bilinear_forward,
        init=bilinear_init,
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

    PyTorchWrapper = registry.get("layers", "PyTorchWrapper.v2")
    model._layers = [
        PyTorchWrapper(
            PyTorchBilinearModel(
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


def bilinear_forward(model: Model, X, is_train: bool):
    return model.layers[0](X, is_train)


def convert_inputs(
    model: Model, X_heads: Tuple[Floats2d, Ints1d], is_train: bool = False
):
    X, H = X_heads

    Xt = xp2torch(X, requires_grad=is_train)
    Ht = xp2torch(H)

    def convert_from_torch_backward(d_inputs: ArgsKwargs) -> Tuple[Floats2d, Ints1d]:
        dX = cast(Floats2d, torch2xp(d_inputs.args[0]))
        return dX, H

    output = ArgsKwargs(args=(Xt, Ht), kwargs={})

    return output, convert_from_torch_backward


def convert_outputs(model, inputs_outputs, is_train):
    _, Y_t = inputs_outputs

    def convert_for_torch_backward(dY: Floats2d) -> ArgsKwargs:
        dY_t = xp2torch(dY)
        return ArgsKwargs(
            args=([Y_t],),
            kwargs={"grad_tensors": [dY_t]},
        )

    Y = cast(Floats2d, torch2xp(Y_t))

    return Y, convert_for_torch_backward
