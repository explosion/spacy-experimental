import numpy as np

from .mst_rust import chu_liu_edmonds

def mst_decode(sent_scores):
    """Apply MST decoding"""

    # Within spacy, a root is encoded as a token that attaches to itself
    # (relative offset 0). However, the decoder uses a specific vertex,
    # typically 0. So, we stub an additional root vertex to accomodate
    # this.

    # We expect a biaffine attention matrix.
    assert sent_scores.shape[0] == sent_scores.shape[1]

    seq_len = sent_scores.shape[0]

    # The MST decoder expects float32, but the input could e.g. be float16.
    sent_scores = sent_scores.astype(np.float32)

    # Create score matrix with root row/column.
    with_root = np.full((seq_len + 1, seq_len + 1), -10000, dtype=sent_scores.dtype)
    with_root[1:, 1:] = sent_scores

    with_root[1:, 0] = sent_scores.diagonal()
    with_root[np.diag_indices(with_root.shape[0])] = -10000

    heads = chu_liu_edmonds(with_root.T, 0)

    # Remove root vertex
    heads = heads[1:]

    for idx, head in enumerate(heads):
        if head == 0:
            heads[idx] = idx
        else:
            heads[idx] = head - 1

    return heads
