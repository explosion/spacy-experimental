import pytest
import spacy
from spacy import util
from spacy.training import Example


@pytest.mark.parametrize("pipe_name", ["experimental_char_tagger_tokenizer", "experimental_char_ner_tokenizer"])
def test_char_tokenizer_overfitting(pipe_name):
    # learn the default English tokenization
    texts = [
        "This is a sentence.",
        "Here is a short, boring sentence.",
        "Here is another!",
    ]
    nlp = spacy.blank("en")
    train_docs = [nlp.make_doc(text) for text in texts]

    nlp = spacy.blank(
        "en",
        vocab=nlp.vocab,
        config={
            "nlp": {
                "tokenizer": {"@tokenizers": "spacy-experimental.char_pretokenizer.v1"}
            }
        },
    )
    nlp.add_pipe(pipe_name)
    train_examples = [Example(nlp.make_doc(doc.text), doc) for doc in train_docs]
    optimizer = nlp.initialize(get_examples=lambda: train_examples)

    for i in range(50):
        nlp.update(train_examples, sgd=optimizer)

    for train_doc, test_doc in zip(train_docs, nlp.pipe(texts)):
        assert train_doc.text == test_doc.text
        assert [t.text for t in train_doc] == [t.text for t in test_doc]
