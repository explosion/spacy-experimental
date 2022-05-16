from spacy.language import Language
from spacy.util import registry
from thinc.api import Config
from thinc.types import Ragged
from spacy_experimental.span_finder.span_finder_component import DEFAULT_CANDIDATES_KEY
from spacy_experimental.span_finder.span_finder_component import span_finder_default_config
)


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


def test_span_finder_component_span_lengths():

    test_min_length = 1
    test_max_length = 4

    nlp = Language()
    doc = nlp(
        "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."
    )
    span_finder = nlp.add_pipe(
        "experimental_span_finder",
        config={"max_length": test_max_length, "min_length": test_min_length},
    )
    nlp.initialize()
    span_finder.set_annotations([doc], span_finder.predict([doc]))

    assert doc.spans[DEFAULT_CANDIDATES_KEY]
    assert len(
        doc.spans[DEFAULT_CANDIDATES_KEY][0]
    ) >= test_min_length and test_max_length >= len(
        doc.spans[DEFAULT_CANDIDATES_KEY][0]
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
