from spacy.util import registry
import spacy
from spacy.tokens import Doc


def test_ngram_subtree_suggester():

    nlp = spacy.blank("en")
    text = "I decided to go for a little run."
    heads = [1, 1, 3, 1, 3, 7, 7, 4, 1]
    deps = ["nsubj", "ROOT", "aux", "xcomp", "prep", "det", "amod", "pobj", "punct"]

    tokenized = nlp(text)
    spaces = [bool(t.whitespace_) for t in tokenized]
    doc = Doc(
        tokenized.vocab,
        words=[t.text for t in tokenized],
        spaces=spaces,
        heads=heads,
        deps=deps,
    )

    suggester = registry.misc.get("spacy-experimental.ngram_subtree_suggester.v1")([1])
    candidates = suggester([doc])

    assert len(candidates.data) == 17


def test_subtree_suggester():

    nlp = spacy.blank("en")
    text = "I decided to go for a little run."
    heads = [1, 1, 3, 1, 3, 7, 7, 4, 1]
    deps = ["nsubj", "ROOT", "aux", "xcomp", "prep", "det", "amod", "pobj", "punct"]

    tokenized = nlp(text)
    spaces = [bool(t.whitespace_) for t in tokenized]
    doc = Doc(
        tokenized.vocab,
        words=[t.text for t in tokenized],
        spaces=spaces,
        heads=heads,
        deps=deps,
    )

    suggester = registry.misc.get("spacy-experimental.subtree_suggester.v1")()
    candidates = suggester([doc])

    assert len(candidates.data) == 15


def test_ngram_chunk_suggester():

    nlp = spacy.blank("en")
    text = "The best thing about visiting the President is the food! I must've drank me fifteen Dr.Peppers."
    heads = [2, 2, 7, 2, 3, 6, 4, 7, 9, 7, 7, 14, 14, 14, 14, 14, 18, 18, 14, 14]
    deps = [
        "det",
        "amod",
        "nsubj",
        "prep",
        "pcomp",
        "det",
        "dobj",
        "ROOT",
        "det",
        "attr",
        "punct",
        "nsubj",
        "aux",
        "aux",
        "ROOT",
        "dative",
        "nummod",
        "compound",
        "dobj",
        "punct",
    ]
    pos = [
        "DET",
        "ADJ",
        "NOUN",
        "ADP",
        "VERB",
        "DET",
        "PROPN",
        "AUX",
        "DET",
        "NOUN",
        "PUNCT",
        "PRON",
        "AUX",
        "AUX",
        "VERB",
        "PRON",
        "NUM",
        "PROPN",
        "PROPN",
        "PUNCT",
    ]

    tokenized = nlp(text)
    spaces = [bool(t.whitespace_) for t in tokenized]
    doc = Doc(
        tokenized.vocab,
        words=[t.text for t in tokenized],
        spaces=spaces,
        heads=heads,
        deps=deps,
        pos=pos,
    )

    suggester = registry.misc.get("spacy-experimental.ngram_chunk_suggester.v1")([1])
    candidates = suggester([doc])

    assert len(candidates.data) == 24


def test_chunk_suggester():

    nlp = spacy.blank("en")
    text = "The best thing about visiting the President is the food! I must've drank me fifteen Dr.Peppers."
    heads = [2, 2, 7, 2, 3, 6, 4, 7, 9, 7, 7, 14, 14, 14, 14, 14, 18, 18, 14, 14]
    deps = [
        "det",
        "amod",
        "nsubj",
        "prep",
        "pcomp",
        "det",
        "dobj",
        "ROOT",
        "det",
        "attr",
        "punct",
        "nsubj",
        "aux",
        "aux",
        "ROOT",
        "dative",
        "nummod",
        "compound",
        "dobj",
        "punct",
    ]
    pos = [
        "DET",
        "ADJ",
        "NOUN",
        "ADP",
        "VERB",
        "DET",
        "PROPN",
        "AUX",
        "DET",
        "NOUN",
        "PUNCT",
        "PRON",
        "AUX",
        "AUX",
        "VERB",
        "PRON",
        "NUM",
        "PROPN",
        "PROPN",
        "PUNCT",
    ]

    tokenized = nlp(text)
    spaces = [bool(t.whitespace_) for t in tokenized]
    doc = Doc(
        tokenized.vocab,
        words=[t.text for t in tokenized],
        spaces=spaces,
        heads=heads,
        deps=deps,
        pos=pos,
    )

    suggester = registry.misc.get("spacy-experimental.chunk_suggester.v1")()
    candidates = suggester([doc])

    assert len(candidates.data) == 6


def test_ngram_sentence_suggester():

    nlp = spacy.blank("en")
    text = "The first sentence. The second sentence. And the third sentence."
    sents = [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]

    tokenized = nlp(text)
    spaces = [bool(t.whitespace_) for t in tokenized]
    doc = Doc(
        tokenized.vocab,
        words=[t.text for t in tokenized],
        spaces=spaces,
        sent_starts=sents,
    )

    suggester = registry.misc.get("spacy-experimental.ngram_sentence_suggester.v1")([1])
    candidates = suggester([doc])

    assert len(candidates.data) == 16


def test_sentence_suggester():

    nlp = spacy.blank("en")
    text = "The first sentence. The second sentence. And the third sentence."
    sents = [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]

    tokenized = nlp(text)
    spaces = [bool(t.whitespace_) for t in tokenized]
    doc = Doc(
        tokenized.vocab,
        words=[t.text for t in tokenized],
        spaces=spaces,
        sent_starts=sents,
    )

    suggester = registry.misc.get("spacy-experimental.sentence_suggester.v1")()
    candidates = suggester([doc])

    assert len(candidates.data) == 3
