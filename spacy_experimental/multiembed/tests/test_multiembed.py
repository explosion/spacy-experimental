import spacy
from typing import List
from spacy.tokens import Doc


multiembed = spacy.registry.architectures.get("spacy-experimental.MultiEmbed.v1")
TEST_TEXTS = [
    "Hello my name is Bob",
    "I know right?",
    "I'm just an example"
]


def create_tables(docs: List[Doc], unk: int):
    orths = {token.orth for doc in docs for token in doc}
    shapes = {token.shape for doc in docs for token in doc}
    tables = {"ORTH": {}, "SHAPE": {}}
    for i, orth in enumerate(orths):
        idx = i if orth != unk else i + 1
        tables["ORTH"][idx] = orth
    for i, shape in enumerate(shapes):
        idx = i if orth != unk else i + 1
        tables["SHAPE"][idx] = shape
    return tables



def test_init_and_serialize():
    nlp = spacy.blank("en")
    docs = list(nlp.pipe(TEST_TEXTS))
    embedder = multiembed(width=10, unk=3)
    tables = create_tables(docs, unk=3)
    embedder.attrs["tables"] = tables
    embedder.initialize(docs)
    embedder.to_disk("test")
    embedder = multiembed(width=10, unk=3)
    embedder.from_disk("test")
test_init_and_serialize()
