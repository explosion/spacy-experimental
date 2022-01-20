import pytest
from spacy import util
from spacy.lang.en import English
from spacy.language import Language
from spacy.training import Example


pytest.importorskip("torch")


TRAIN_DATA = [
    (
        "She likes green eggs",
        {
            "heads": [1, 1, 3, 1],
            "deps": ["nsubj", "ROOT", "amod", "dobj"],
            "sent_starts": [1, 0, 0, 0],
        },
    ),
    (
        "Eat blue ham",
        {
            "heads": [0, 2, 0],
            "deps": ["ROOT", "amod", "dobj"],
            "sent_starts": [1, 0, 0],
        },
    ),
]

PARTIAL_DATA = [
    (
        "She likes green eggs",
        {
            "heads": [1, 1, 3, 1],
            "deps": ["nsubj", "ROOT", "amod", "dobj"],
            "sent_starts": [1, 0, 0, 0],
        },
    ),
    # Misaligned partial annotation.
    (
        "Eat blue ham",
        {
            "words": ["Ea", "t", "blue", "ham"],
            "heads": [0, 0, 3, 0],
            "deps": ["ROOT", "", "amod", "dobj"],
            "sent_starts": [1, 0, 0, 0],
        },
    ),
]


def test_initialize_examples():
    nlp = Language()
    nlp.add_pipe("sentencizer")
    nlp.add_pipe("experimental_arc_predicter")
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    # you shouldn't really call this more than once, but for testing it should be fine
    nlp.initialize(get_examples=lambda: train_examples)
    with pytest.raises(TypeError):
        nlp.initialize(get_examples=lambda: None)
    with pytest.raises(TypeError):
        nlp.initialize(get_examples=lambda: train_examples[0])
    with pytest.raises(TypeError):
        nlp.initialize(get_examples=lambda: [])
    with pytest.raises(TypeError):
        nlp.initialize(get_examples=train_examples)


def test_incomplete_data():
    nlp = English.from_config()
    nlp.add_pipe("sentencizer")
    nlp.add_pipe("experimental_arc_predicter")
    train_examples = []
    for t in PARTIAL_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))

    optimizer = nlp.initialize(get_examples=lambda: train_examples)

    for i in range(150):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses, annotates=["sentencizer"])
    assert losses["experimental_arc_predicter"] < 0.00001

    test_text = "She likes green eggs"
    doc = nlp(test_text)
    assert doc[0].head == doc[1]
    assert doc[1].head == doc[1]
    assert doc[2].head == doc[3]
    assert doc[3].head == doc[1]


def test_overfitting_IO():
    nlp = English.from_config()
    nlp.add_pipe("sentencizer")
    nlp.add_pipe("experimental_arc_predicter")
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))

    optimizer = nlp.initialize(get_examples=lambda: train_examples)

    for i in range(150):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses, annotates=["sentencizer"])
    assert losses["experimental_arc_predicter"] < 0.00001

    test_text = "She likes green eggs"
    doc = nlp(test_text)
    assert doc[0].head == doc[1]
    assert doc[1].head == doc[1]
    assert doc[2].head == doc[3]
    assert doc[3].head == doc[1]

    # Check model after a {to,from}_disk roundtrip
    with util.make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
        doc2 = nlp2(test_text)
        assert doc2[0].head == doc2[1]
        assert doc2[1].head == doc2[1]
        assert doc2[2].head == doc2[3]
        assert doc2[3].head == doc2[1]

    # Check model after a {to,from}_bytes roundtrip
    nlp_bytes = nlp.to_bytes()
    nlp3 = English()
    nlp3.add_pipe("sentencizer")
    nlp3.add_pipe("experimental_arc_predicter")
    nlp3.from_bytes(nlp_bytes)
    doc3 = nlp3(test_text)
    assert doc3[0].head == doc3[1]
    assert doc3[1].head == doc3[1]
    assert doc3[2].head == doc3[3]
    assert doc3[3].head == doc3[1]
