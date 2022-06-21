from typing import Optional, Iterable, cast
from thinc.api import get_current_ops, Ops
from thinc.types import Ragged, Ints1d
from spacy.tokens import Doc
from spacy.pipeline.spancat import Suggester


def build_span_finder_suggester(candidates_key: str) -> Suggester:
    """Suggest every candidate predicted by the SpanFinder"""

    def span_finder_suggester(
        docs: Iterable[Doc], *, ops: Optional[Ops] = None
    ) -> Ragged:
        if ops is None:
            ops = get_current_ops()
        spans = []
        lengths = []
        for doc in docs:
            length = 0
            if doc.spans[candidates_key]:
                for span in doc.spans[candidates_key]:
                    spans.append([span.start, span.end])
                    length += 1

            lengths.append(length)

        lengths_array = cast(Ints1d, ops.asarray(lengths, dtype="i"))
        if len(spans) > 0:
            output = Ragged(ops.asarray(spans, dtype="i"), lengths_array)
        else:
            output = Ragged(ops.xp.zeros((0, 0), dtype="i"), lengths_array)

        return output

    return span_finder_suggester
