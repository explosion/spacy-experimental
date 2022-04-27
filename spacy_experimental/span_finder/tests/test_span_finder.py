from spacy.language import Language
from spacy.util import registry
from thinc.api import Config
from thinc.types import Ragged
import spacy_experimental.span_finder.span_finder_component as span_finder

SPAN_KEY = "pytest"


def test_span_finder_model():
    nlp = Language()

    docs = [nlp("This is an example."), nlp("This is the second example.")]
    docs[0].spans[SPAN_KEY] = [docs[0][3:4]]
    docs[1].spans[SPAN_KEY] = [docs[1][3:5]]

    total_tokens = 0
    for doc in docs:
        total_tokens += len(doc)

    config = Config().from_str(span_finder.span_finder_default_config).interpolate()
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
