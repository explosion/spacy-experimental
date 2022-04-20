from spacy.language import Language
from spacy.util import registry
import spacy

SPAN_KEY = "pytest"


def test_subtree_suggester():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(
        "This is a super great interesting example for testing the subtree suggester."
    )
    suggester = registry.misc.get("experimental.subtree_suggester.v1")([1])
    candidates = suggester([doc])

    print(candidates)
