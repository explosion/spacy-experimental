from typing import List, Tuple, Dict
from thinc.types import Ints1d, Ints2d, Floats2d
from thinc.api import NumpyOps
from spacy.language import Language
from spacy.tokens import Doc

# type alias to make writing this less tedious
MentionClusters = List[List[Tuple[int, int]]]

DEFAULT_CLUSTER_PREFIX = "coref_clusters"
DEFAULT_CLUSTER_HEAD_PREFIX = "coref_head_clusters"


@Language.factory(
    "experimental_span_cleaner",
    assigns=["doc.spans"],
    default_config={"prefix": DEFAULT_CLUSTER_HEAD_PREFIX},
)
def make_span_cleaner(nlp: Language, name: str, *, prefix: str) -> "SpanCleaner":
    """Create a span cleaner component.

    Given a prefix, a span cleaner removes any spans on the Doc where the key
    matches the prefix.
    """

    return SpanCleaner(prefix)


class SpanCleaner:
    def __init__(self, prefix: str):
        self.prefix = prefix

    def __call__(self, doc: Doc) -> Doc:
        for key in list(doc.spans.keys()):
            if key.startswith(self.prefix):
                del doc.spans[key]
        return doc


def matches_coref_prefix(prefix: str, key: str) -> bool:
    """Check if a span key matches a coref prefix.

    Given prefix "xxx", "xxx_1" is a matching span, but "xxx_yyy" and
    "xxx_yyy_1" are not matching spans. The prefix must only be followed by an
    underscore and an integer.
    """
    if not key.startswith(prefix):
        return False

    # remove the "prefix_" bit
    suffix = key[len(prefix) + 1 :]
    try:
        int(suffix)
    except ValueError:
        return False

    return True


def get_sentence_ids(doc: Doc) -> List[int]:
    """Given a Doc, return a list of the sentence ID of each token,
    where the sentence ID is the index of the sentence in the Doc.

    Used in coref to make sure mentions don't cross sentence boundaries.
    """
    out = []
    sent_id = -1
    for tok in doc:
        if tok.is_sent_start:
            sent_id += 1
        out.append(sent_id)
    return out


# from model.py, refactored to be non-member
def get_predicted_antecedents(xp, antecedent_idx: Ints2d, antecedent_scores: Floats2d):
    """Get the ID of the antecedent for each span. -1 if no antecedent."""
    predicted_antecedents = xp.argmax(antecedent_scores, axis=1) - 1
    out = xp.full(antecedent_idx.shape[0], -1, dtype=antecedent_idx.dtype)
    if predicted_antecedents.max() == -1:
        return out
    valid_indices = predicted_antecedents != -1
    valid_preds = antecedent_idx[
        xp.arange(antecedent_idx.shape[0]), predicted_antecedents
    ][valid_indices]
    xp.place(
        out,
        valid_indices,
        valid_preds,
    )
    return out


# from model.py, refactored to be non-member
def get_predicted_clusters(
    xp,
    span_starts: Ints1d,
    span_ends: Ints1d,
    antecedent_idx: Ints2d,
    antecedent_scores: Floats2d,
):
    """Convert predictions to usable cluster data.

    return values:

    clusters: a list of spans (i, j) that are a cluster

    Note that not all spans will be in the final output; spans with no
    antecedent or referrent are omitted from clusters and mention2cluster.
    """
    # Get predicted antecedents
    ops = NumpyOps()
    predicted_antecedents = ops.asarray(
        get_predicted_antecedents(xp, antecedent_idx, antecedent_scores)
    ).tolist()

    # Get predicted clusters
    mention_to_cluster_id = {}
    predicted_clusters = []
    for i, predicted_idx in enumerate(predicted_antecedents):
        if predicted_idx < 0:
            continue
        assert i > predicted_idx, f"span idx: {i}; antecedent idx: {predicted_idx}"
        # Check antecedent's cluster
        antecedent = (int(span_starts[predicted_idx]), int(span_ends[predicted_idx]))
        antecedent_cluster_id = mention_to_cluster_id.get(antecedent, -1)
        if antecedent_cluster_id == -1:
            antecedent_cluster_id = len(predicted_clusters)
            predicted_clusters.append([antecedent])
            mention_to_cluster_id[antecedent] = antecedent_cluster_id
        # Add mention to cluster
        mention = (int(span_starts[i]), int(span_ends[i]))
        predicted_clusters[antecedent_cluster_id].append(mention)
        mention_to_cluster_id[mention] = antecedent_cluster_id

    predicted_clusters = [tuple(c) for c in predicted_clusters]
    return predicted_clusters


def create_head_span_idxs(ops, doclen: int):
    """Helper function to create single-token span indices."""
    aa = ops.xp.arange(0, doclen)
    bb = ops.xp.arange(0, doclen) + 1
    return ops.asarray2i([aa, bb]).T


def get_clusters_from_doc(
    doc: Doc, *, use_heads: bool = False, prefix: str = None
) -> List[List[Tuple[int, int]]]:
    """Convert the span clusters in a Doc to simple integer tuple lists. The
    ints are char spans, to be tokenization independent.

    If `use_heads` is True, then the heads (roots) of the spans will be used.

    If a `prefix` is provided, then only spans matching the prefix will be used.
    """
    out = []
    keys = sorted(list(doc.spans.keys()))
    for key in keys:
        if prefix is not None and not matches_coref_prefix(prefix, key):
            continue

        val = doc.spans[key]
        cluster = []

        for span in val:
            if use_heads:
                head_i = span.root.i
                head = doc[head_i]
                char_span = (head.idx, head.idx + len(head))
            else:
                char_span = (span[0].idx, span[-1].idx + len(span[-1]))

            cluster.append(char_span)

        # don't want duplicates
        cluster = list(set(cluster))
        out.append(cluster)
    return out


def create_gold_scores(
    ments: Ints2d, clusters: List[List[Tuple[int, int]]]
) -> Floats2d:
    """Given mentions considered for antecedents and gold clusters,
    construct a gold score matrix. This does not include the placeholder.

    In the gold matrix, the value of a true antecedent is True, and otherwise
    it is False. These will represented as 1/0 values.
    """
    # make a mapping of mentions to cluster id
    # id is not important but equality will be
    ment2cid: Dict[Tuple[int, int], int] = {}
    for cid, cluster in enumerate(clusters):
        for ment in cluster:
            ment2cid[ment] = cid

    ll = len(ments)
    ops = NumpyOps()
    cpu_ments = ops.asarray(ments)

    out = ops.alloc2f(ll, ll)
    for ii, ment in enumerate(cpu_ments):
        cid = ment2cid.get((int(ment[0]), int(ment[1])))
        if cid is None:
            # this is not in a cluster so it has no antecedent
            continue

        # this might change if no real antecedent is a candidate
        for jj, ante in enumerate(cpu_ments):
            # antecedents must come first
            if jj >= ii:
                break
            if cid == ment2cid.get((int(ante[0]), int(ante[1])), -1):
                out[ii, jj] = 1.0

    return out
