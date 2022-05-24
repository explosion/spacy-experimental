from spacy.language import Language
from spacy.util import registry
from thinc.api import Config
from thinc.types import Ragged
from spacy_experimental.span_finder.span_finder_component import (
    DEFAULT_CANDIDATES_KEY,
    span_finder_default_config,
)
import pytest

SPAN_KEY = "pytest"


def test_span_finder_model():
    nlp = Language()

    docs = [nlp("This is an example."), nlp("This is the second example.")]
    docs[0].spans[SPAN_KEY] = [docs[0][3:4]]
    docs[1].spans[SPAN_KEY] = [docs[1][3:5]]

    total_tokens = 0
    for doc in docs:
        total_tokens += len(doc)

    config = Config().from_str(span_finder_default_config).interpolate()
    model = registry.resolve(config)["model"]

    model.initialize(X=docs)
    predictions = model.predict(docs)

    assert len(predictions) == total_tokens
    assert len(predictions[0]) == 2


def test_span_finder_component():
    nlp = Language()

    docs = [nlp("This is an example."), nlp("This is the second example.")]
    docs[0].spans[SPAN_KEY] = [docs[0][3:4]]
    docs[1].spans[SPAN_KEY] = [docs[1][3:5]]

    span_finder = nlp.add_pipe("experimental_span_finder")
    nlp.initialize()
    span_finder.set_annotations(docs, span_finder.predict(docs))

    assert docs[0].spans["span_candidates"]


@pytest.mark.parametrize(
    "min_length, max_length, span_count", [(0, 0, 8), (2, 0, 6), (0, 1, 2), (2, 3, 2)]
)
def test_set_annotations_span_lengths(min_length, max_length, span_count):
    nlp = Language()
    doc = nlp("Me and Jenny goes together like peas and carrots.")
    span_finder = nlp.add_pipe(
        "experimental_span_finder",
        config={"max_length": max_length, "min_length": min_length},
    )
    nlp.initialize()
    # Starts    [Me, Jenny, peas]
    # Ends      [Jenny, peas, carrots]
    scores = [
        (1, 0),
        (0, 0),
        (1, 1),
        (0, 0),
        (0, 0),
        (0, 0),
        (1, 1),
        (0, 0),
        (0, 1),
        (0, 0),
    ]
    span_finder.set_annotations([doc], scores)

    assert doc.spans[DEFAULT_CANDIDATES_KEY]
    assert len(doc.spans[DEFAULT_CANDIDATES_KEY]) == span_count

    # Assert below will fail when max_length is set to 0
    if max_length <= 0:
        max_length = 1000

    assert all(
        min_length <= len(span) <= max_length
        for span in doc.spans[DEFAULT_CANDIDATES_KEY]
    )


def test_span_finder_suggester():

    nlp = Language()
    docs = [nlp("This is an example."), nlp("This is the second example.")]
    docs[0].spans[SPAN_KEY] = [docs[0][3:4]]
    docs[1].spans[SPAN_KEY] = [docs[1][3:5]]
    span_finder = nlp.add_pipe("experimental_span_finder")
    nlp.initialize()
    span_finder.set_annotations(docs, span_finder.predict(docs))

    suggester = registry.misc.get("spacy-experimental.span_finder_suggester.v1")(
        candidates_key="span_candidates"
    )

    candidates = suggester(docs)

    span_length = 0
    for doc in docs:
        span_length += len(doc.spans["span_candidates"])

    assert span_length == len(candidates.dataXd)
    assert type(candidates) == Ragged
    assert len(candidates.dataXd[0]) == 2