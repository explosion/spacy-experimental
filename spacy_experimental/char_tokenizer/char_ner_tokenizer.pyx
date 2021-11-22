# cython: infer_types=True, profile=True, binding=True
from typing import Optional, Callable

import srsly
from thinc.api import Model, SequenceCategoricalCrossentropy, Config

from spacy.tokens.doc cimport Doc

from spacy.pipeline.ner import EntityRecognizer
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.tokens import Doc as PyDoc, Span
from spacy.training import Example
from spacy.util import registry

from .scorers import tokenizer_score


default_model_config = """
[model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = true
nO = null

[model.tok2vec]
@architectures = "spacy.Tok2Vec.v2"

[model.tok2vec.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = 128
attrs = ["ORTH","LOWER","IS_DIGIT","IS_ALPHA","IS_SPACE","IS_PUNCT"]
rows = [1000,500,50,50,50,50]
include_static_vectors = false

[model.tok2vec.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 128
depth = 4
window_size = 4
maxout_pieces = 2
"""
DEFAULT_CHAR_NER_TOKENIZER_MODEL = Config().from_str(default_model_config)["model"]


@Language.factory(
    "experimental_char_ner_tokenizer",
    assigns=["token.is_sent_start"],
    default_config={"model": DEFAULT_CHAR_NER_TOKENIZER_MODEL, "scorer": {"@scorers": "spacy-experimental.tokenizer_scorer.v1"}},
    default_score_weights={"token_f": 0.5, "token_p": 0.0, "token_r": 0.0, "token_acc": None},
    retokenizes=True,
)
def make_char_ner_tokenizer(nlp: Language, name: str, model: Model, scorer: Optional[Callable]):
    return CharNERTokenizer(nlp.vocab, model, name, scorer=scorer)


class CharNERTokenizer(EntityRecognizer):
    def __init__(
        self,
        vocab,
        model,
        name="exp_char_ner_tokenizer",
        *,
        scorer=tokenizer_score,
    ):
        super().__init__(vocab, model, name, None, scorer=scorer)

    def set_annotations(self, docs, scores):
        super().set_annotations(docs, scores)
        cdef Doc doc
        cdef int j, k, end
        for doc in docs:
            orig_text = doc.text
            # merge all tokens marked as ents with trailing space if present
            with doc.retokenize() as retokenizer:
                for ent in doc.ents:
                    end = ent.end
                    if ent.end < len(doc) and doc[ent.end].text == " ":
                        end += 1
                    retokenizer.merge(doc[ent.start:end])
            # try to merge any remaining trailing SPACY spaces
            with doc.retokenize() as retokenizer:
                j = 0
                while j < doc.length - 1:
                    if doc[j + 1].text == " " and \
                            not doc[j].is_space and \
                            not doc[j].text.endswith(" "):
                        retokenizer.merge(doc[j:j+2])
                        j += 1
                    j += 1
            doc.set_ents([], default="missing")
            j = 0
            while j < doc.length:
                text = doc[j].text
                if len(text) > 1 and text.endswith(" ") and not text.isspace():
                    lex = doc.vocab.get(doc.vocab.mem, text[:-1])
                    doc.c[j].lex = lex
                    doc.c[j].spacy = True
                j += 1
            assert doc.text == orig_text

    def update(self, examples, *, **kwargs):
        examples = [Example(example.predicted, self._convert_to_char_doc(example.reference)) for example in examples]
        super().update(examples, **kwargs)

    def initialize(self, get_examples, *, **kwargs):
        examples = [Example(example.predicted, self._convert_to_char_doc(example.reference)) for example in get_examples()]
        super().initialize(lambda: examples, **kwargs)

    def _convert_to_char_doc(self, doc: Doc):
        words = list(doc.text)
        char_doc = PyDoc(doc.vocab, words=words, spaces=[False] * len(words))
        if not doc.has_unknown_spaces:
            char_doc.ents = [
                Span(char_doc, token.idx, token.idx+len(token.text), label="TOKEN")
                for token in doc
            ]
        return char_doc
