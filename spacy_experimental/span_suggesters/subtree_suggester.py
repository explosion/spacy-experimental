from typing import List, Optional, Iterable, cast
from thinc.api import get_current_ops, Ops
from thinc.types import Ragged, Ints1d
from spacy.pipeline.spancat import Suggester
from spacy.tokens import Doc
from spacy.util import registry
from .merge_suggesters import merge_suggestions


def build_ngram_subtree_suggester(sizes: List[int]) -> Suggester:
    """Suggest ngrams and subtrees. Requires annotations from the DependencyParser"""

    ngram_suggester = registry.misc.get("spacy.ngram_suggester.v1")(sizes)

    def ngram_subtree_suggester(
        docs: Iterable[Doc], *, ops: Optional[Ops] = None
    ) -> Ragged:
        ngram_suggestions = ngram_suggester(docs, ops=ops)
        subtree_suggestions = subtree_suggester(docs, ops=ops)
        return merge_suggestions([ngram_suggestions, subtree_suggestions], ops=ops)

    return ngram_subtree_suggester


def build_subtree_suggester() -> Suggester:
    """Suggest subtrees. Requires annotations from the DependencyParser"""
    return subtree_suggester


def subtree_suggester(docs: Iterable[Doc], *, ops: Optional[Ops] = None) -> Ragged:
    if ops is None:
        ops = get_current_ops()

    spans = []
    lengths = []

    for doc in docs:
        cache = set()
        length = 0

        for token in doc:
            if (token.left_edge.i, token.i + 1) not in cache:
                spans.append((token.left_edge.i, token.i + 1))
                cache.add((token.left_edge.i, token.i + 1))
                length += 1
            if (token.i, token.right_edge.i + 1) not in cache:
                spans.append((token.i, token.right_edge.i + 1))
                cache.add((token.i, token.right_edge.i + 1))
                length += 1
            if (token.left_edge.i, token.right_edge.i + 1) not in cache:
                spans.append((token.left_edge.i, token.right_edge.i + 1))
                cache.add((token.left_edge.i, token.right_edge.i + 1))
                length += 1

        lengths.append(length)

    lengths_array = cast(Ints1d, ops.asarray(lengths, dtype="i"))
    if len(spans) > 0:
        output = Ragged(ops.asarray(spans, dtype="i"), lengths_array)
    else:
        output = Ragged(ops.xp.zeros((0, 0), dtype="i"), lengths_array)

    return output
