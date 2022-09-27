from typing import List, Tuple, cast
import warnings

from thinc.api import Model, chain, tuplify, get_width
from thinc.api import PyTorchWrapper, ArgsKwargs
from thinc.types import Floats2d, Ints1d
from thinc.util import torch, xp2torch, torch2xp

from spacy.tokens import Doc
from .coref_util import get_sentence_ids, MentionClusters, matches_coref_prefix

try:
    from .pytorch_span_resolver_model import SpanResolverModel
except ImportError:
    SpanResolverModel = None


def build_span_resolver(
    tok2vec: Model[List[Doc], List[Floats2d]],
    hidden_size: int,
    distance_embedding_size: int,
    conv_channels: int,
    window_size: int,
    max_distance: int,
    prefix: str,
) -> Model[List[Doc], List[MentionClusters]]:
    if SpanResolverModel is None:
        raise ImportError("SpanResolver requires PyTorch: pip install thinc[torch]")

    nI = None

    with Model.define_operators({">>": chain, "&": tuplify}):
        span_resolver: Model[List[Floats2d], List[Floats2d]] = Model(
            "span_resolver",
            forward=span_resolver_forward,
            init=span_resolver_init,
            dims={"nI": nI},
            attrs={
                "distance_embedding_size": distance_embedding_size,
                "hidden_size": hidden_size,
                "conv_channels": conv_channels,
                "window_size": window_size,
                "max_distance": max_distance,
            },
        )
        head_info = build_get_head_metadata(prefix)
        model = (tok2vec & head_info) >> span_resolver
        model.set_ref("span_resolver", span_resolver)

    return model


def span_resolver_init(model: Model, X=None, Y=None):
    if model.layers:
        return

    if X is not None and model.has_dim("nI") is None:
        model.set_dim("nI", get_width(X))

    hidden_size = model.attrs["hidden_size"]
    distance_embedding_size = model.attrs["distance_embedding_size"]
    conv_channels = model.attrs["conv_channels"]
    window_size = model.attrs["window_size"]
    max_distance = model.attrs["max_distance"]

    model._layers = [
        PyTorchWrapper(
            SpanResolverModel(
                model.get_dim("nI"),
                hidden_size,
                distance_embedding_size,
                conv_channels,
                window_size,
                max_distance,
            ),
            convert_inputs=convert_span_resolver_inputs,
        )
    ]


def span_resolver_forward(model: Model, X, is_train: bool):
    return model.layers[0](X, is_train)


def convert_span_resolver_inputs(
    model: Model,
    X: Tuple[List[Floats2d], Tuple[List[Ints1d], List[Ints1d]]],
    is_train: bool,
):
    tok2vec, (sent_ids, head_ids) = X

    # Normally we should use the input is_train, but for these two it's not relevant
    def backprop(args: ArgsKwargs) -> Tuple[List[Floats2d], None]:
        gradients = cast(Floats2d, torch2xp(args.args[1]))
        # The sent_ids and head_ids are None because no gradients
        return ([gradients], None)

    word_features = xp2torch(tok2vec[0], requires_grad=is_train)
    sent_ids_tensor = xp2torch(sent_ids[0], requires_grad=False)
    if not head_ids[0].size:
        head_ids_tensor = torch.empty(size=(0,))
    else:
        head_ids_tensor = xp2torch(head_ids[0], requires_grad=False)

    argskwargs = ArgsKwargs(
        args=(sent_ids_tensor, word_features, head_ids_tensor), kwargs={}
    )
    return argskwargs, backprop


def build_get_head_metadata(prefix):
    model = Model(
        "HeadDataProvider", attrs={"prefix": prefix}, forward=head_data_forward
    )
    return model


def head_data_forward(model, docs, is_train):
    """A layer to generate the extra data needed for the span resolver."""
    sent_ids = []
    head_ids = []
    prefix = model.attrs["prefix"]
    for doc in docs:
        sids = model.ops.asarray2i(get_sentence_ids(doc))
        sent_ids.append(sids)
        heads = []
        for key, sg in doc.spans.items():
            if not matches_coref_prefix(prefix, key):
                continue

            for span in sg:
                heads.append(span[0].i)
                if len(span) > 1:
                    # TODO assign number to warning
                    # Besides simple errors, this can be caused by tokenization
                    # mismatches.
                    warnings.warn(
                        f"Input span has length {len(span)}, but should be 1."
                    )
        heads = model.ops.asarray2i(heads)
        head_ids.append(heads)
    # Each of these is a list with one entry per doc.
    # Backprop is just a placeholder, since the input is docs.
    return (sent_ids, head_ids), lambda x: []
