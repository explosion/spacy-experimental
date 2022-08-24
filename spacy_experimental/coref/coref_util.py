from typing import List, Tuple, Dict, cast
from thinc.types import Ints2d
import srsly
from spacy.language import Language
from spacy.tokens import Doc
import spacy.util as util

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
        self.cfg: Dict[str, Any] = {"prefix": prefix}

    def __call__(self, doc: Doc) -> Doc:
        prefix = self.cfg["prefix"]
        for key in list(doc.spans.keys()):
            if key.startswith(prefix):
                del doc.spans[key]
        return doc

    def to_bytes(self, **kwargs):
        serializers = {
            "cfg": lambda: srsly.json_dumps(self.cfg),
        }
        return util.to_bytes(serializers, [])

    def from_bytes(self, data, **kwargs):
        deserializers = {
            "cfg": lambda b: self.cfg.update(srsly.json_loads(b)),
        }
        util.from_bytes(data, deserializers, [])
        return self

    def to_disk(self, path, **kwargs):
        path = util.ensure_path(path)
        serializers = {
            "cfg": lambda p: srsly.write_json(p, self.cfg),
        }
        return util.to_disk(path, serializers, [])

    def from_disk(self, path, **kwargs):
        path = util.ensure_path(path)
        serializers = {
            "cfg": lambda p: self.cfg.update(srsly.read_json(p)),
        }
        util.from_disk(path, serializers, [])


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
def get_predicted_antecedents(xp, antecedent_idx, antecedent_scores):
    """Get the ID of the antecedent for each span. -1 if no antecedent."""
    predicted_antecedents = []
    for i, idx in enumerate(xp.argmax(antecedent_scores, axis=1) - 1):
        if idx < 0:
            predicted_antecedents.append(-1)
        else:
            predicted_antecedents.append(antecedent_idx[i][idx])
    return predicted_antecedents


# from model.py, refactored to be non-member
def get_predicted_clusters(
    xp, span_starts, span_ends, antecedent_idx, antecedent_scores
):
    """Convert predictions to usable cluster data.

    return values:

    clusters: a list of spans (i, j) that are a cluster

    Note that not all spans will be in the final output; spans with no
    antecedent or referrent are omitted from clusters and mention2cluster.
    """
    # Get predicted antecedents
    predicted_antecedents = get_predicted_antecedents(
        xp, antecedent_idx, antecedent_scores
    )

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


def select_non_crossing_spans(
    idxs: List[int], starts: List[int], ends: List[int], limit: int
) -> List[int]:
    """Given a list of spans sorted in descending order, return the indexes of
    spans to keep, discarding spans that cross.

    Nested spans are allowed.
    """
    # ported from Model._extract_top_spans
    selected: List[int] = []
    start_to_max_end: Dict[int, int] = {}
    end_to_min_start: Dict[int, int] = {}

    for idx in idxs:
        if len(selected) >= limit or idx > len(starts):
            break

        start, end = starts[idx], ends[idx]
        cross = False

        for ti in range(start, end):
            max_end = start_to_max_end.get(ti, -1)
            if ti > start and max_end > end:
                cross = True
                break

            min_start = end_to_min_start.get(ti, -1)
            if ti < end and 0 <= min_start < start:
                cross = True
                break

        if not cross:
            # this index will be kept
            # record it so we can exclude anything that crosses it
            selected.append(idx)
            max_end = start_to_max_end.get(start, -1)
            if end > max_end:
                start_to_max_end[start] = end
            min_start = end_to_min_start.get(end, -1)
            if min_start == -1 or start < min_start:
                end_to_min_start[end] = start

    # sort idxs by order in doc
    selected = sorted(selected, key=lambda idx: (starts[idx], ends[idx]))
    # This was causing many repetitive entities in the output - removed for now
    # while len(selected) < limit:
    #     selected.append(selected[0])  # this seems a bit weird?
    return selected


def create_head_span_idxs(ops, doclen: int):
    """Helper function to create single-token span indices."""
    aa = ops.xp.arange(0, doclen)
    bb = ops.xp.arange(0, doclen) + 1
    return ops.asarray2i([aa, bb]).T


def get_clusters_from_doc(doc) -> List[List[Tuple[int, int]]]:
    """Convert the span clusters in a Doc to simple integer tuple lists. The
    ints are char spans, to be tokenization independent.
    """
    out = []
    keys = sorted(list(doc.spans.keys()))
    for key in keys:
        val = doc.spans[key]
        cluster = []
        for span in val:

            head_i = span.root.i
            head = doc[head_i]
            char_span = (head.idx, head.idx + len(head))
            cluster.append(char_span)

        # don't want duplicates
        cluster = list(set(cluster))
        out.append(cluster)
    return out


def create_gold_scores(
    ments: Ints2d, clusters: List[List[Tuple[int, int]]]
) -> List[List[bool]]:
    """Given mentions considered for antecedents and gold clusters,
    construct a gold score matrix. This does not include the placeholder.

    In the gold matrix, the value of a true antecedent is True, and otherwise
    it is False. These will be converted to 1/0 values later.
    """
    # make a mapping of mentions to cluster id
    # id is not important but equality will be
    ment2cid: Dict[Tuple[int, int], int] = {}
    for cid, cluster in enumerate(clusters):
        for ment in cluster:
            ment2cid[ment] = cid

    ll = len(ments)
    out = []
    # The .tolist() call is necessary with cupy but not numpy
    mentuples = [cast(Tuple[int, int], tuple(mm.tolist())) for mm in ments]
    for ii, ment in enumerate(mentuples):
        if ment not in ment2cid:
            # this is not in a cluster so it has no antecedent
            out.append([False] * ll)
            continue

        # this might change if no real antecedent is a candidate
        row = []
        cid = ment2cid[ment]
        for jj, ante in enumerate(mentuples):
            # antecedents must come first
            if jj >= ii:
                row.append(False)
                continue

            row.append(cid == ment2cid.get(ante, -1))

        out.append(row)

    # caller needs to convert to array, and add placeholder
    return out
