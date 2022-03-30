from typing import Optional, Iterable, cast
from thinc.api import get_current_ops, Ops

from thinc.types import Ragged, Ints1d

from spacy.compat import Protocol, runtime_checkable
from spacy.tokens import Doc
from spacy.util import registry


@runtime_checkable
class Suggester(Protocol):
    def __call__(self, docs: Iterable[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        ...


@registry.misc("spacy-experimental.span_boundary_detection_suggester.v1")
def build_sbd_suggester() -> Suggester:
    """Suggest every candidate predicted by the SBD"""

    def sbd_suggester(docs: Iterable[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        if ops is None:
            ops = get_current_ops()
        spans = []
        lengths = []
        for doc in docs:
            starts = []
            ends = []
            length = 0
            cache = set()
            for token in doc:
                if token._.span_start == 1:
                    starts.append(token.i)
                if token._.span_end == 1:
                    ends.append(token.i + 1)

            for start in starts:
                for end in ends:
                    if start < end and (start, end) not in cache:
                        spans.append([start, end])
                        cache.add((start, end))
                        length += 1

            lengths.append(length)

        lengths_array = cast(Ints1d, ops.asarray(lengths, dtype="i"))
        if len(spans) > 0:
            output = Ragged(ops.asarray(spans, dtype="i"), lengths_array)
        else:
            output = Ragged(ops.xp.zeros((0, 0), dtype="i"), lengths_array)

        return output

    return sbd_suggester
