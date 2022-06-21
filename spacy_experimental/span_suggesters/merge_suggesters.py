from typing import List, Optional, cast
import numpy
from thinc.api import get_current_ops, Ops
from thinc.types import Ragged, Ints1d


def merge_suggestions(suggestions: List[Ragged], ops: Optional[Ops] = None) -> Ragged:
    if ops is None:
        ops = get_current_ops()

    spans = []
    lengths = []

    if len(suggestions) == 0:
        lengths_array = cast(Ints1d, ops.asarray(lengths, dtype="i"))
        return Ragged(ops.xp.zeros((0, 0), dtype="i"), lengths_array)

    len_docs = len(suggestions[0])
    assert all(len_docs == len(x) for x in suggestions)

    for i in range(len_docs):
        combined = ops.xp.vstack([s[i].data for s in suggestions if len(s[i].data) > 0])
        uniqued = numpy.unique(ops.to_numpy(combined), axis=0)
        spans.append(ops.asarray(uniqued))
        lengths.append(uniqued.shape[0])

    lengths_array = cast(Ints1d, ops.asarray(lengths, dtype="i"))
    if len(spans) > 0:
        output = Ragged(ops.xp.vstack(spans), lengths_array)
    else:
        output = Ragged(ops.xp.zeros((0, 0), dtype="i"), lengths_array)

    return output
