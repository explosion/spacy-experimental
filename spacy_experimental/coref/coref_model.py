from typing import List, Tuple, Callable, cast

from thinc.api import Model, chain, get_width
from thinc.api import PyTorchWrapper, ArgsKwargs
from thinc.types import Floats2d, Ints2d
from thinc.util import xp2torch, torch2xp

from spacy.tokens import Doc

try:
    from .pytorch_coref_model import CorefClusterer
except ImportError:
    CorefClusterer = None


def build_coref_model(
    tok2vec: Model[List[Doc], List[Floats2d]],
    distance_embedding_size: int,
    hidden_size: int,
    depth: int,
    dropout: float,
    # pairs to keep per mention after rough scoring
    antecedent_limit: int,
    antecedent_batch_size: int,
) -> Model[List[Doc], Tuple[Floats2d, Ints2d]]:

    if CorefClusterer is None:
        raise ImportError("Coref requires PyTorch: pip install thinc[torch]")

    nI = None

    with Model.define_operators({">>": chain}):
        coref_clusterer: Model[List[Floats2d], Tuple[Floats2d, Ints2d]] = Model(
            "coref_clusterer",
            forward=coref_forward,
            init=coref_init,
            dims={"nI": nI},
            attrs={
                "distance_embedding_size": distance_embedding_size,
                "hidden_size": hidden_size,
                "depth": depth,
                "dropout": dropout,
                "antecedent_limit": antecedent_limit,
                "antecedent_batch_size": antecedent_batch_size,
            },
        )

        model = tok2vec >> coref_clusterer
        model.set_ref("coref_clusterer", coref_clusterer)
    return model


def coref_init(model: Model, X=None, Y=None):
    if model.layers:
        return

    if X is not None and model.has_dim("nI") is None:
        model.set_dim("nI", get_width(X))

    hidden_size = model.attrs["hidden_size"]
    depth = model.attrs["depth"]
    dropout = model.attrs["dropout"]
    antecedent_limit = model.attrs["antecedent_limit"]
    antecedent_batch_size = model.attrs["antecedent_batch_size"]
    distance_embedding_size = model.attrs["distance_embedding_size"]

    model._layers = [
        PyTorchWrapper(
            CorefClusterer(
                model.get_dim("nI"),
                distance_embedding_size,
                hidden_size,
                depth,
                dropout,
                antecedent_limit,
                antecedent_batch_size,
            ),
            convert_inputs=convert_coref_clusterer_inputs,
            convert_outputs=convert_coref_clusterer_outputs,
        )
    ]


def coref_forward(model: Model, X, is_train: bool):
    return model.layers[0](X, is_train)


def convert_coref_clusterer_inputs(model: Model, X_: List[Floats2d], is_train: bool):
    # The input here is List[Floats2d], one for each doc
    # just use the first
    # TODO real batching
    X = X_[0]
    word_features = xp2torch(X, requires_grad=is_train)

    def backprop(args: ArgsKwargs) -> List[Floats2d]:
        # convert to xp and wrap in list
        gradients = cast(Floats2d, torch2xp(args.args[0]))
        return [gradients]

    return ArgsKwargs(args=(word_features,), kwargs={}), backprop


def convert_coref_clusterer_outputs(
    model: Model, inputs_outputs, is_train: bool
) -> Tuple[Tuple[Floats2d, Ints2d], Callable]:
    _, outputs = inputs_outputs
    scores, indices = outputs

    def convert_for_torch_backward(dY: Floats2d) -> ArgsKwargs:
        dY_t = xp2torch(dY[0])
        return ArgsKwargs(
            args=([scores],),
            kwargs={"grad_tensors": [dY_t]},
        )

    scores_xp = cast(Floats2d, torch2xp(scores))
    indices_xp = cast(Ints2d, torch2xp(indices))
    return (scores_xp, indices_xp), convert_for_torch_backward
