from spacy.language import Language
from spacy.util import registry
from thinc.api import Config
from thinc.types import Ragged

SPAN_KEY = "pytest"

DEFAULT_SBD_MODEL_CONFIG = """
[model]
@architectures = "spacy-experimental.span_boundary_detection_model.v1"

[model.scorer]
@layers = "spacy.LinearLogistic.v1"
nO = 2

[model.tok2vec]
@architectures = "spacy.Tok2Vec.v1"

[model.tok2vec.embed]
@architectures = "spacy.MultiHashEmbed.v1"
width = 96
rows = [5000, 2000, 1000, 1000]
attrs = ["ORTH", "PREFIX", "SUFFIX", "SHAPE"]
include_static_vectors = false

[model.tok2vec.encode]
@architectures = "spacy.MaxoutWindowEncoder.v1"
width = ${model.tok2vec.embed.width}
window_size = 1
maxout_pieces = 3
depth = 4
"""


def test_sbd_model():
    nlp = Language()

    docs = [nlp("This is an example."), nlp("This is the second example.")]
    docs[0].spans[SPAN_KEY] = [docs[0][3:4]]
    docs[1].spans[SPAN_KEY] = [docs[1][3:5]]

    total_tokens = 0
    for doc in docs:
        total_tokens += len(doc)

    config = Config().from_str(DEFAULT_SBD_MODEL_CONFIG).interpolate()
    model = registry.resolve(config)["model"]

    model.initialize(X=docs)
    predictions = model.predict(docs)

    assert len(predictions) == total_tokens
    assert len(predictions[0]) == 2


def test_sbd_component():
    nlp = Language()

    docs = [nlp("This is an example."), nlp("This is the second example.")]
    docs[0].spans[SPAN_KEY] = [docs[0][3:4]]
    docs[1].spans[SPAN_KEY] = [docs[1][3:5]]

    total_tokens = 0
    for doc in docs:
        total_tokens += len(doc)

    sbd = nlp.add_pipe("spacy-experimental_span_boundary_detection_component_v1")
    nlp.initialize()
    scores = sbd.predict(docs)

    assert len(scores) == total_tokens
    assert len(scores[0]) == 2


def test_sbd_suggester():
    nlp = Language()
    nlp.add_pipe("spacy-experimental_span_boundary_detection_component_v1")
    nlp.initialize()
    suggester = registry.misc.get(
        "spacy-experimental.span_boundary_detection_suggester.v1"
    )()

    docs = [nlp("This is an example."), nlp("This is the second example.")]

    candidates = suggester(docs)

    assert type(candidates) == Ragged
    assert len(candidates.dataXd[0]) == 2
