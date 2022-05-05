from spacy.util import registry
import spacy


def test_ngram_subtree_suggester():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("I decided to go for a little run.")
    suggester = registry.misc.get("experimental.ngram_subtree_suggester.v1")([1])
    candidates = suggester([doc])

    assert len(candidates.data) == 17


def test_subtree_suggester():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("I decided to go for a little run.")
    suggester = registry.misc.get("experimental.subtree_suggester.v1")()
    candidates = suggester([doc])

    assert len(candidates.data) == 15


def test_ngram_chunk_suggester():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(
        "The best thing about visiting the President is the food! I must've drank me fifteen Dr.Peppers."
    )
    suggester = registry.misc.get("experimental.ngram_chunk_suggester.v1")([1])
    candidates = suggester([doc])

    assert len(candidates.data) == 24


def test_chunk_suggester():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(
        "The best thing about visiting the President is the food! I must've drank me fifteen Dr.Peppers."
    )
    suggester = registry.misc.get("experimental.chunk_suggester.v1")()
    candidates = suggester([doc])

    assert len(candidates.data) == 6


def test_ngram_sentence_suggester():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("The first sentence. The second sentence. And the third sentence.")
    suggester = registry.misc.get("experimental.ngram_sentence_suggester.v1")([1])
    candidates = suggester([doc])

    assert len(candidates.data) == 16


def test_sentence_suggester():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("The first sentence. The second sentence. And the third sentence.")
    suggester = registry.misc.get("experimental.sentence_suggester.v1")()
    candidates = suggester([doc])

    assert len(candidates.data) == 3
