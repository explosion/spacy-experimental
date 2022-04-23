from typing import List, Optional, Iterable, cast
from thinc.api import get_current_ops, Ops
from thinc.types import Ragged, Ints1d
from spacy.compat import Protocol, runtime_checkable
from spacy.tokens import Doc
from spacy.util import registry


@runtime_checkable
class Suggester(Protocol):
    def __call__(self, docs: Iterable[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        ...


@registry.misc("experimental.subtree_suggester.v1")
def build_subtree_suggester(sizes: List[int]) -> Suggester:
    """Suggest ngrams and subtrees. Requires annotations from the DependencyParser"""

    def subtree_suggester(docs: Iterable[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        if ops is None:
            ops = get_current_ops()
        spans = []
        lengths = []

        if sizes:
            suggester = registry.misc.get("spacy.ngram_suggester.v1")(sizes)

        for doc in docs:
            cache = set()
            length = 0

            # ngram-suggestion
            if sizes:
                ngram_spans = suggester([doc], ops=ops)
                for ngram_span in ngram_spans.data:
                    ngram_span = ops.to_numpy(ngram_span)
                    spans.append((ngram_span[0], ngram_span[1]))
                    cache.add((ngram_span[0], ngram_span[1]))
                    length += 1

            # subtree-suggestion
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

    return subtree_suggester
