import pickle
import pytest
from spacy import util
from spacy.lang.en import English
from spacy.language import Language
from spacy.training import Example


TRAIN_DATA = [
    ("She likes green eggs", {"lemmas": ["she", "like", "green", "egg"]}),
    ("Eat blue ham", {"lemmas": ["eat", "blue", "ham"]}),
]

PARTIAL_DATA = [
    # partial annotation
    ("She likes green eggs", {"lemmas": ["", "like", "green", ""]}),
    # misaligned partial annotation
    (
        "He hates green eggs",
        {
            "words": ["He", "hat", "es", "green", "eggs"],
            "lemmas": ["", "hat", "e", "green", ""],
        },
    ),
]


def test_initialize_examples():
    nlp = Language()
    lemmatizer = nlp.add_pipe("experimental_edit_tree_lemmatizer")
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


def test_initialize_from_labels():
    nlp = Language()
    lemmatizer = nlp.add_pipe("experimental_edit_tree_lemmatizer")
    lemmatizer.min_tree_freq = 1
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    nlp.initialize(get_examples=lambda: train_examples)

    nlp2 = Language()
    lemmatizer2 = nlp2.add_pipe("experimental_edit_tree_lemmatizer")
    lemmatizer2.initialize(
        get_examples=lambda: train_examples,
        labels=lemmatizer.label_data,
    )
    assert lemmatizer2.tree2label == {1: 0, 3: 1, 4: 2, 6: 3}


def test_no_data():
    # Test that the lemmatizer provides a nice error when there's no tagging data / labels
    TEXTCAT_DATA = [
        ("I'm so happy.", {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}}),
        ("I'm so angry", {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}}),
    ]
    nlp = English()
    nlp.add_pipe("experimental_edit_tree_lemmatizer")
    nlp.add_pipe("textcat")

    train_examples = []
    for t in TEXTCAT_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))

    with pytest.raises(ValueError):
        nlp.initialize(get_examples=lambda: train_examples)


def test_incomplete_data():
    # Test that the lemmatizer works with incomplete information
    nlp = English()
    lemmatizer = nlp.add_pipe("experimental_edit_tree_lemmatizer")
    lemmatizer.min_tree_freq = 1
    train_examples = []
    for t in PARTIAL_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    optimizer = nlp.initialize(get_examples=lambda: train_examples)
    for i in range(50):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    assert losses["experimental_edit_tree_lemmatizer"] < 0.00001

    # test the trained model
    test_text = "She likes blue eggs"
    doc = nlp(test_text)
    assert doc[1].lemma_ == "like"
    assert doc[2].lemma_ == "blue"


def test_overfitting_IO():
    nlp = English()
    lemmatizer = nlp.add_pipe("experimental_edit_tree_lemmatizer")
    lemmatizer.min_tree_freq = 1
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))

    optimizer = nlp.initialize(get_examples=lambda: train_examples)

    for i in range(50):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    assert losses["experimental_edit_tree_lemmatizer"] < 0.00001

    test_text = "She likes blue eggs"
    doc = nlp(test_text)
    assert doc[0].lemma_ == "she"
    assert doc[1].lemma_ == "like"
    assert doc[2].lemma_ == "blue"
    assert doc[3].lemma_ == "egg"

    # Check model after a {to,from}_disk roundtrip
    with util.make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
        doc2 = nlp2(test_text)
        assert doc2[0].lemma_ == "she"
        assert doc2[1].lemma_ == "like"
        assert doc2[2].lemma_ == "blue"
        assert doc2[3].lemma_ == "egg"

    # Check model after a {to,from}_bytes roundtrip
    nlp_bytes = nlp.to_bytes()
    nlp3 = English()
    nlp3.add_pipe("experimental_edit_tree_lemmatizer")
    nlp3.from_bytes(nlp_bytes)
    doc3 = nlp3(test_text)
    assert doc3[0].lemma_ == "she"
    assert doc3[1].lemma_ == "like"
    assert doc3[2].lemma_ == "blue"
    assert doc3[3].lemma_ == "egg"

    # Check model after a pickle roundtrip.
    nlp_bytes = pickle.dumps(nlp)
    nlp4 = pickle.loads(nlp_bytes)
    doc4 = nlp4(test_text)
    assert doc4[0].lemma_ == "she"
    assert doc4[1].lemma_ == "like"
    assert doc4[2].lemma_ == "blue"
    assert doc4[3].lemma_ == "egg"


def test_lemmatizer_requires_labels():
    nlp = English()
    nlp.add_pipe("experimental_edit_tree_lemmatizer")
    with pytest.raises(ValueError):
        nlp.initialize()


def test_lemmatizer_label_data():
    nlp = English()
    lemmatizer = nlp.add_pipe("experimental_edit_tree_lemmatizer")
    lemmatizer.min_tree_freq = 1
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))

    nlp.initialize(get_examples=lambda: train_examples)

    nlp2 = English()
    lemmatizer2 = nlp2.add_pipe("experimental_edit_tree_lemmatizer")
    lemmatizer2.initialize(
        get_examples=lambda: train_examples, labels=lemmatizer.label_data
    )

    # Verify that the labels and trees are the same.
    assert lemmatizer.labels == lemmatizer2.labels
    assert lemmatizer.trees.to_bytes() == lemmatizer2.trees.to_bytes()
