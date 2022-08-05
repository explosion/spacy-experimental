import spacy
import thinc
import srsly
from pathlib import Path

from spacy.tokens import Doc
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from thinc.api import Model, chain, with_array
from thinc.api import list2ragged, ragged2list, concatenate, noop
from thinc.types import Floats2d, DTypes, Ints2d, Array2d
from typing import Optional, Callable, List, Dict, Any
from typing import Union, Sequence, Tuple

Embed = thinc.registry.layers.get("Embed.v1")
Maxout = thinc.registry.layers.get("Maxout.v1")
Dropout = thinc.registry.layers.get("Dropout.v1")
Vectors = thinc.registry.layers.get("spacy.StaticVectors.v2")
Extract = thinc.registry.layers.get("spacy.FeatureExtractor.v1")


InT = Union[Sequence[Any], Array2d]
OutT = Ints2d


@spacy.registry.callbacks("set_attr")
def create_callback(
    path: Path,
    component: str,
    attr: str,
    layer: Optional[str],
) -> Callable[[Language], Language]:
    """
    Should be set as a callback of [initialize.before_init].
    You need to set the right ref in your model when you create it.
    This is useful when you have some layer that requires a data
    file from disk. The value will only be loaded during the '
    initialize' step before training.
    After training the attribute value will be serialized into the model,
    and then during deserialization it's loaded
    back in with the model data.
    """
    attr_value = srsly.read_msgpack(path)

    def set_attr(nlp: Language) -> Language:
        if not nlp.has_pipe(component):
            raise ValueError(
                "Trying to set attribute for non-existing component"
            )
        pipe: TrainablePipe = nlp.get_pipe(component)
        model = None
        for lay in list(pipe.model.walk()):
            if lay.name == layer:
                model = lay
                break
        if model is None:
            raise ValueError(
                f"Haven't found {layer} in component {component}."
            )
        model.attrs[attr] = attr_value
        return nlp
    return set_attr


@thinc.registry.layers("remap_ids.v2")
def remap_ids(
    table: Dict[Any, int] = {},
    default: int = 0,
    dtype: DTypes = "i",
    column: Optional[int] = None
) -> Model[InT, OutT]:
    """
    Customizes the remap_ids layer from thinc. This
    code is here temporarily until
    https://github.com/explosion/thinc/pull/726
    gets merged.
    """
    return Model(
        "remap_ids",
        remap_forward,
        attrs={
            "table": table,
            "dtype": dtype,
            "default": default,
            "column": column
        },
    )


def remap_forward(
    model: Model[InT, OutT], inputs: InT, is_train: bool
) -> Tuple[OutT, Callable]:
    table = model.attrs["table"]
    default = model.attrs["default"]
    dtype = model.attrs["dtype"]
    column = model.attrs["column"]
    if column is not None:
        inputs = inputs[:, column]
    # We wrap int around x, because in cupy each integer
    # in the arrays is a cuda array with shape ()
    values = [table.get(int(x), default) for x in inputs]
    arr = model.ops.asarray2i(values, dtype=dtype)
    output = model.ops.reshape2i(arr, -1, 1)

    def backprop(dY: OutT) -> InT:
        return model.ops.asarray([])

    return output, backprop


def _make_embed(
    attr: str,
    unk: int,
    width: int,
    column: int,
    table: Dict[int, int],
    dropout: float
) -> Model[List[Doc], Floats2d]:
    """
    Helper function to create an embedding layer.
    """
    rows = len(table) + 1
    embedder = chain(
        remap_ids(table, default=unk, column=column),
        Embed(nO=width, nV=rows, column=0, dropout=dropout)
    )
    return embedder


@spacy.registry.architectures("spacy.MultiEmbed.v1")
def MultiEmbed(
    attrs: List[str],
    width: int,
    unk: int,
    *,
    tables: Optional[Dict[str, Dict[int, int]]] = None,
    include_static_vectors: Optional[bool] = False,
    dropout: Optional[float] = 0,
) -> Model[List[Doc], Floats2d]:
    model_attrs = {
        "tables": tables,
        "unk": unk,
        "include_static_vectors": include_static_vectors,
        "attrs": attrs,
        "dropout": dropout
    }
    layers = []
    # Create dummy embedding layer to be materialized at init
    embedders = [chain(remap_ids(), Embed(column=0)) for x in attrs]
    embedder_stack = chain(
        Extract(attrs),
        list2ragged(),
        with_array(concatenate(*embedders)),
    )
    if include_static_vectors:
        embedding_layer = chain(
            concatenate(
                embedder_stack, Vectors(width)
            ),
            Dropout(rate=dropout)
        )
    else:
        embedding_layer = embedder_stack
    layers.append(embedding_layer)
    # Build proper output layer
    full_width = (len(attrs) + include_static_vectors) * width
    max_out = chain(
        with_array(
            Maxout(
                width, full_width, nP=3, dropout=dropout, normalize=True
            )
        ),
        ragged2list()
    )
    layers.append(max_out)
    model: Model = Model(
        "multiembed",
        forward,
        init=init,
        attrs=model_attrs,
        dims={"width": width},
        layers=layers,
        params={},
    )
    model.set_ref("embedder-stack", embedder_stack)
    model.set_ref("output-layer", max_out)
    return model


def forward(
        model: Model[List[Doc], List[Floats2d]],
        X: List[Doc],
        is_train=False
) -> Floats2d:
    embedding_layer = model.layers[0]
    output_layer = model.layers[1]
    embedded, bp_embed = embedding_layer(X, is_train)
    Y, bp_output = output_layer(embedded, is_train)

    def backprop(dY: List[Floats2d]):
        dO = bp_output(dY)
        dX = bp_embed(dO)
        return dX

    return Y, backprop


def init(
    model: Model[List[Doc], Floats2d],
    X: Optional[List[Doc]] = None,
    Y: Optional = None,
) -> None:
    """
    Build and initialize the embedding and output
    all layers of MultiEmbed and initialize.
    """
    width = model.get_dim("width")
    tables = model.attrs["tables"]
    if tables is None:
        raise ValueError(
            "tables have to be set before initialization"
        )
    unk = model.attrs["unk"]
    attrs = model.attrs["attrs"]
    dummy_embedder_stack = model.get_ref("embedder-stack")
    output_layer = model.get_ref("output-layer")

    if set(attrs).union(tables.keys()) != set(tables.keys()):
        extra = set(attrs) - set(tables.keys())
        raise ValueError(
            f"Could not find provided attribute(s) {extra} "
            f"in tables: {list(tables.keys())}"
        )
    dropout = model.attrs["dropout"]
    embedders = []
    for i, attr in enumerate(attrs):
        mapper = tables[attr]
        embedding = _make_embed(
            attr, unk, width, i, mapper, dropout
        )
        embedders.append(embedding)
    embedder_stack = chain(
        Extract(attrs),
        list2ragged(),
        with_array(concatenate(*embedders)),
    )
    model.replace_node(dummy_embedder_stack, embedder_stack)
    embedding_layer = model.layers[0]
    embedding_layer.initialize(X)
    embedded, _ = embedding_layer(X, is_train=False)
    output_layer.initialize(embedded)
