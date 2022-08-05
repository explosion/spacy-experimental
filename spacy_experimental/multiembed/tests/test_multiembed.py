import os
import math

import pytest
import spacy
import srsly

from spacy.training import Example
from spacy import util
from typing import List
from spacy.tokens import Doc
from thinc.api import Config


multiembed = spacy.registry.architectures.get(
    "spacy-experimental.MultiEmbed.v1"
)
set_attr = spacy.registry.misc.get(
    "spacy-experimental.set_attr.v1"
)

TEST_TEXTS = [
    "Hello my name is Bob",
    "I know right?",
    "I'm just an example"
]
TEST_LABELS = [
    {"cats": {"greeting": 1.0}},
    {"cats": {"question": 1.0}},
    {"cats": {"statement": 1.0}}
]


cfg_string_textcat = """
    [nlp]
    lang = "en"
    pipeline = ["tok2vec","textcat"]

    [components]

    [components.textcat]
    factory = "textcat"

    [components.textcat.model]
    @architectures = "spacy.TextCatCNN.v2"
    exclusive_classes = true

    [components.textcat.model.tok2vec]
    @architectures = "spacy.Tok2VecListener.v1"
    width = ${components.tok2vec.model.encode.width}

    [components.tok2vec]
    factory = "tok2vec"

    [components.tok2vec.model]
    @architectures = "spacy.Tok2Vec.v2"

    [components.tok2vec.model.embed]
    @architectures = "spacy-experimental.MultiEmbed.v1"
    width = ${components.tok2vec.model.encode.width}
    attrs = ["ORTH", "SHAPE"]
    include_static_vectors = false
    unk = 0

    [components.tok2vec.model.encode]
    @architectures = "spacy.MaxoutWindowEncoder.v2"
    width = 96
    depth = 1
    window_size = 1
    maxout_pieces = 2

    [initialize]

    [initialize.before_init]
    @callbacks = "set_attr.v1"
    path = tables.msg
    component = "tok2vec"
    layer = "multiembed"
    attr = "tables"
    """


def _get_examples():
    nlp = spacy.blank("en")
    for text, label in zip(TEST_TEXTS, TEST_LABELS):
        yield Example.from_dict(nlp.make_doc(text), label)


def create_tables(docs: List[Doc], unk: int):
    orths = {token.orth for doc in docs for token in doc}
    shapes = {token.shape for doc in docs for token in doc}
    tables = {"ORTH": {}, "SHAPE": {}}
    for i, orth in enumerate(orths):
        idx = i if orth != unk else i + 1
        tables["ORTH"][orth] = idx
    for i, shape in enumerate(shapes):
        idx = i if orth != unk else i + 1
        tables["SHAPE"][orth] = idx
    return tables


def test_init_and_serialize():
    nlp = spacy.blank("en")
    docs = list(nlp.pipe(TEST_TEXTS))
    embedder = multiembed(["ORTH", "SHAPE"], width=10, unk=3)
    tables = create_tables(docs, unk=3)
    embedder.attrs["tables"] = tables
    embedder.initialize(docs)
    embedder.to_disk("test")
    embedder = multiembed(["ORTH", "SHAPE"], width=10, unk=3)
    embedder.from_disk("test")
    os.remove("test")


def test_tables_not_set_error():
    nlp = spacy.blank("en")
    docs = list(nlp.pipe(TEST_TEXTS))
    embedder = multiembed(["ORTH", "SHAPE"], width=10, unk=3)
    with pytest.raises(ValueError, match="tables have to be set"):
        embedder.initialize(docs)


def test_unknown_attribute_error():
    nlp = spacy.blank("en")
    docs = list(nlp.pipe(TEST_TEXTS))
    embedder = multiembed(["ORTH", "SHAPE", "EXTRA"], width=10, unk=3)
    tables = create_tables(docs, unk=3)
    embedder.attrs["tables"] = tables
    with pytest.raises(ValueError, match="Could not find"):
        embedder.initialize(docs)


def test_overfit():
    textcat_config = Config().from_str(cfg_string_textcat)
    spacy.registry.callbacks.register("set_attr.v1", func=set_attr)
    nlp = util.load_model_from_config(
        textcat_config, auto_fill=True, validate=True
    )
    docs = [nlp.make_doc(text) for text in TEST_TEXTS]
    tables = create_tables(docs, unk=0)
    srsly.write_msgpack("tables.msg", tables)
    examples = list(_get_examples())
    optimizer = nlp.initialize(_get_examples)
    optimizer.learn_rate = 1
    for i in range(10):
        losses = {}
        nlp.update(
            examples, sgd=optimizer, losses=losses
        )
    os.remove("tables.msg")
    assert math.isclose(losses['textcat'], 0.0)
