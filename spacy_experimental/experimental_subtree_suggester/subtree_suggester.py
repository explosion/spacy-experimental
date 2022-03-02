from typing import Optional, Iterable, Any, cast
from thinc.api import get_current_ops, Ops
from thinc.types import Ragged, Ints1d

from spacy.compat import Protocol, runtime_checkable
from spacy.tokens import Doc
from spacy.util import registry


@runtime_checkable
class Suggester(Protocol):
    def __call__(self, docs: Iterable[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        ...


@registry.misc("spacy-experimental.subtree_suggester.v1")
def build_subtree_suggester() -> Suggester:
    """Suggest every connected subtree per token"""

    def subtree_suggester(docs: Iterable[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        if ops is None:
            ops = get_current_ops()
        spans = []
        lengths = []
        for doc in docs:
            length = 0
            cache = {}
            for token in doc:

                # ngram-1
                if (token.i, token.i + 1) not in cache:
                    spans.append([token.i, token.i + 1])
                    cache[(token.i, token.i + 1)] = True
                    length += 1

                # ngram-2
                # if token.i + 2 <= len(doc):
                #    if (token.i, token.i + 2) not in cache:
                #        spans.append([token.i, token.i + 2])
                #        cache[(token.i, token.i + 2)] = True
                #       length += 1

                # left-edge to token
                if token.left_edge.i < token.i + 1:
                    if (token.left_edge.i, token.i + 1) not in cache:
                        spans.append([token.left_edge.i, token.i + 1])
                        cache[(token.left_edge.i, token.i + 1)] = True
                        length += 1

                # token to right-edge
                if token.i < token.right_edge.i + 1:
                    if (token.i, token.right_edge.i + 1) not in cache:
                        spans.append([token.i, token.right_edge.i + 1])
                        cache[(token.i, token.right_edge.i + 1)] = True
                        length += 1

                # left-edge to right-edge
                if token.left_edge.i < token.right_edge.i + 1:
                    if (token.left_edge.i, token.right_edge.i + 1) not in cache:
                        spans.append([token.left_edge.i, token.right_edge.i + 1])
                        cache[(token.left_edge.i, token.right_edge.i + 1)] = True
                        length += 1

            lengths.append(length)

        lengths_array = cast(Ints1d, ops.asarray(lengths, dtype="i"))
        if len(spans) > 0:
            output = Ragged(ops.asarray(spans, dtype="i"), lengths_array)
        else:
            output = Ragged(ops.xp.zeros((0, 0), dtype="i"), lengths_array)

        return output

    return subtree_suggester
