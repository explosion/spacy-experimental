from typing import List, Optional, Iterable, cast
from thinc.api import get_current_ops, Ops
from thinc.types import Ragged, Ints1d
from spacy.tokens import Doc
from spacy.util import registry
from spacy.pipeline.spancat import Suggester
from .merge_suggesters import merge_suggestions


def build_ngram_sentence_suggester(sizes: List[int]) -> Suggester:
    """Suggest ngrams and sentences. Requires sentence boundaries."""
    ngram_suggester = registry.misc.get("spacy.ngram_suggester.v1")(sizes)

    def ngram_sentence_suggester(
        docs: Iterable[Doc], *, ops: Optional[Ops] = None
    ) -> Ragged:
        ngram_suggestions = ngram_suggester(docs, ops=ops)
        sentence_suggestions = sentence_suggester(docs, ops=ops)
        return merge_suggestions([ngram_suggestions, sentence_suggestions], ops=ops)

    return ngram_sentence_suggester


def build_sentence_suggester() -> Suggester:
    """Suggest sentences. Requires sentence boundaries."""
    return sentence_suggester


def sentence_suggester(docs: Iterable[Doc], *, ops: Optional[Ops] = None) -> Ragged:
    if ops is None:
        ops = get_current_ops()
    spans = []
    lengths = []

    for doc in docs:
        cache = set()
        length = 0

        for sentence in doc.sents:
            if (sentence.start, sentence.end) not in cache:
                spans.append((sentence.start, sentence.end))
                cache.add((sentence.start, sentence.end))
                length += 1

        lengths.append(length)

    lengths_array = cast(Ints1d, ops.asarray(lengths, dtype="i"))
    if len(spans) > 0:
        output = Ragged(ops.asarray(spans, dtype="i"), lengths_array)
    else:
        output = Ragged(ops.xp.zeros((0, 0), dtype="i"), lengths_array)

    return output
