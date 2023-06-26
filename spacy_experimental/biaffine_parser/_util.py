from thinc.api import get_array_module
from thinc.types import Ints1d


def lengths2offsets(lens: Ints1d) -> Ints1d:
    """Convert an array of lengths to an array of offsets. For instance,
    the array [6, 3, 5] is converted into the offsets [0, 6, 9]."""
    xp = get_array_module(lens)
    starts_ends = xp.empty(len(lens) + 1, dtype="i")
    starts_ends[0] = 0
    lens.cumsum(out=starts_ends[1:])
    return starts_ends[:-1]
