from thinc.types import Floats2d, Floats1d
from typing import List
from thinc.api import Model, chain, with_array
from spacy.tokens import Doc
from numpy import float32


def build_finder_model(
    tok2vec: Model[List[Doc], List[Floats2d]], scorer: Model[Floats2d, Floats2d]
) -> Model[List[Doc], Floats2d]:

    logistic_layer = with_array(scorer)

    model = chain(tok2vec, logistic_layer, flattener())

    model.set_ref("tok2vec", tok2vec)
    model.set_ref("scorer", scorer)
    model.set_ref("logistic_layer", logistic_layer)

    return model


def flattener() -> Model[Floats1d, Floats1d]:
    """Flattens the input to a 1-dimensional list of scores"""

    def forward(
        model: Model[Floats1d, Floats1d], X: Floats1d, is_train: bool
    ) -> Floats1d:
        output = []
        lengths = []

        for doc in X:
            lengths.append(len(doc))
            for token in doc:
                output.append(token)
        output = model.ops.asarray(output, dtype=float32)

        def backprop(Y) -> Floats2d:
            offset = 0
            original = []
            for length in lengths:
                original.append(Y[offset : offset + length])
                offset += length
            return original

        return output, backprop

    return Model("Flattener", forward=forward)
