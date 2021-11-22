# cython: infer_types=True, profile=True, binding=True
from typing import Optional, Callable

import srsly
from thinc.api import Model, SequenceCategoricalCrossentropy, Config

from spacy.tokens.doc cimport Doc

from spacy.pipeline.tagger import Tagger
from spacy.language import Language
from spacy.errors import Errors
from spacy.scorer import Scorer
from spacy.tokens import Doc as PyDoc
from spacy.training import Example
from spacy.util import registry, get_words_and_spaces

from .scorers import tokenizer_senter_score


default_model_config = """
[model]
@architectures = "spacy.Tagger.v1"

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
DEFAULT_CHAR_TAGGER_TOKENIZER_MODEL = Config().from_str(default_model_config)["model"]


@Language.factory(
    "experimental_char_tagger_tokenizer",
    assigns=[],
    default_config={"model": DEFAULT_CHAR_TAGGER_TOKENIZER_MODEL, "annotate_sents": True, "scorer": {"@scorers": "spacy-experimental.tokenizer_senter_scorer.v1"}},
    default_score_weights={"token_f": 0.5, "token_p": 0.0, "token_r": 0.0, "token_acc": None, "sents_f": 1.0, "sents_p": 0.0, "sents_r": 0.0},
    retokenizes=True,
)
def make_char_tagger_tokenizer(nlp: Language, name: str, model: Model, annotate_sents: bool, scorer: Optional[Callable]):
    return CharTaggerTokenizer(nlp.vocab, model, name, annotate_sents=annotate_sents, scorer=scorer)


class CharTaggerTokenizer(Tagger):
    def __init__(
        self,
        vocab,
        model,
        name="char_tagger_tokenizer",
        *,
        annotate_sents=True,
        scorer=tokenizer_senter_score,
    ):
        super().__init__(vocab, model, name, overwrite=True, scorer=scorer)
        self.cfg["annotate_sents"] = annotate_sents

    def set_annotations(self, docs, batch_tag_ids):
        if isinstance(docs, Doc):
            docs = [docs]
        cdef Doc doc
        cdef int idx, token_start
        labels = self.labels
        sent_label_id = -1
        if "S" in labels:
            sent_label_id = labels.index("S")
        token_label_id = labels.index("T")
        for i, doc in enumerate(docs):
            orig_text = doc.text
            doc_tag_ids = batch_tag_ids[i]
            if hasattr(doc_tag_ids, "get"):
                doc_tag_ids = doc_tag_ids.get()
            sent_starts = set()
            if self.cfg["annotate_sents"]:
                sent_starts = {j for j, tag_id in enumerate(doc_tag_ids) if tag_id == sent_label_id}
                # first character always starts a sentence
                sent_starts.add(0)
            token_start = 0
            with doc.retokenize() as retokenizer:
                for j, tag_id in enumerate(doc_tag_ids[1:], start=1):
                    if tag_id == token_label_id or tag_id == sent_label_id:
                        retokenizer.merge(doc[token_start:j])
                        token_start = j
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
            # reset all the tag values
            j = 0
            while j < doc.length:
                text = doc[j].text
                if len(text) > 1 and text.endswith(" "):
                    lex = doc.vocab.get(doc.vocab.mem, text[:-1])
                    doc.c[j].lex = lex
                    doc.c[j].spacy = True
                if self.cfg["annotate_sents"]:
                    if doc.c[j].idx in sent_starts:
                        doc.c[j].sent_start = 1
                    else:
                        doc.c[j].sent_start = -1
                doc.c[j].tag = 0
                j += 1
            assert doc.text == orig_text

    def update(self, examples, *, **kwargs):
        examples = [Example(example.predicted, self._convert_to_char_doc(example.reference)) for example in examples]
        super().update(examples, **kwargs)

    def initialize(self, get_examples, *, **kwargs):
        examples = [Example(example.predicted, self._convert_to_char_doc(example.reference)) for example in get_examples()]
        super().initialize(lambda: examples, **kwargs)

    def _convert_to_char_doc(self, doc: Doc):
        """Convert a doc to character tokenization with TAG values for the
        tokenization and sentence segmentation (if enabled):
        - S: character starts a sentence
        - T: character starts a token (overridden by S if enabled)
        - I: character is within a token
        - O: character is not part of a token
        """
        words = list(doc.text)
        char_doc = PyDoc(doc.vocab, words=words, spaces=[False] * len(words))
        if not doc.has_unknown_spaces:
            for char_token in char_doc:
                char_token.tag_ = "O"
            for token in doc:
                char_doc[token.idx].tag_ = "T"
                if self.cfg["annotate_sents"]:
                    if token.is_sent_start:
                        char_doc[token.idx].tag_ = "S"
                for idx in range(token.idx + 1, token.idx + len(token.text)):
                    char_doc[idx].tag_ = "I"
        return char_doc
