from typing import List, Optional, Iterable, cast
from thinc.api import get_current_ops, Ops
from thinc.types import Ragged, Ints1d
from spacy.tokens import Doc
from spacy.util import registry
from spacy.pipeline.spancat import Suggester
from .merge_suggesters import merge_suggestions


def build_ngram_chunk_suggester(sizes: List[int]) -> Suggester:
    """Suggest ngrams and noun chunks. Requires annotations from the Tagger & DependencyParser"""
    ngram_suggester = registry.misc.get("spacy.ngram_suggester.v1")(sizes)

    def ngram_chunk_suggester(
        docs: Iterable[Doc], *, ops: Optional[Ops] = None
    ) -> Ragged:
        ngram_suggestions = ngram_suggester(docs, ops=ops)
        chunk_suggestions = chunk_suggester(docs, ops=ops)
        return merge_suggestions([ngram_suggestions, chunk_suggestions], ops=ops)

    return ngram_chunk_suggester


def build_chunk_suggester() -> Suggester:
    """Suggest noun chunks. Requires annotations from the Tagger & DependencyParser"""
    return chunk_suggester


def chunk_suggester(docs: Iterable[Doc], *, ops: Optional[Ops] = None) -> Ragged:
    if ops is None:
        ops = get_current_ops()
    spans = []
    lengths = []

    for doc in docs:
        cache = set()
        length = 0

        for noun_chunk in doc.noun_chunks:
            if (noun_chunk.start, noun_chunk.end) not in cache:
                spans.append((noun_chunk.start, noun_chunk.end))
                cache.add((noun_chunk.start, noun_chunk.end))
                length += 1

        lengths.append(length)

    lengths_array = cast(Ints1d, ops.asarray(lengths, dtype="i"))
    if len(spans) > 0:
        output = Ragged(ops.asarray(spans, dtype="i"), lengths_array)
    else:
        output = Ragged(ops.xp.zeros((0, 0), dtype="i"), lengths_array)

    return output
