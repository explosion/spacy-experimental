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
def build_subtree_suggester(ngram_sizes: List[int]) -> Suggester:
    """Suggest every set ngram and subtree. Requires annotations from the DependencyParser"""

    def subtree_suggester(docs: Iterable[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        if ops is None:
            ops = get_current_ops()
        spans = []
        lengths = []
        for doc in docs:
            cache = set()
            length = 0

            if ngram_sizes:
                starts = ops.xp.arange(len(doc), dtype="i")
                starts = starts.reshape((-1, 1))
                length = 0
                for size in ngram_sizes:
                    if size <= len(doc):
                        starts_size = starts[: len(doc) - (size - 1)]
                        spans.append(ops.xp.hstack((starts_size, starts_size + size)))
                        length += spans[-1].shape[0]
                print(spans)
                for span in spans:
                    cache.add(span)

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
