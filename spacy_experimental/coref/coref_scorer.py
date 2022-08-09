from typing import Iterable, List, Tuple
from statistics import mean

from spacy.tokens import Doc
from spacy.training import Example

from .coref_util import DEFAULT_CLUSTER_PREFIX
from .coref_util import matches_coref_prefix


def score_coref_clusters(
    examples: Iterable[Example],
    *,
    span_cluster_prefix: str = DEFAULT_CLUSTER_PREFIX,
    **kwargs,
):
    """Score a batch of examples using LEA.

    For details on how LEA works and why to use it see the paper:
    Which Coreference Evaluation Metric Do You Trust? A Proposal for a Link-based Entity Aware Metric
    Moosavi and Strube, 2016
    https://api.semanticscholar.org/CorpusID:17606580
    """

    evaluator = ClusterEvaluator(lea)

    for ex in examples:
        p_clusters = doc2clusters(ex.predicted, span_cluster_prefix)
        g_clusters = doc2clusters(ex.reference, span_cluster_prefix)
        cluster_info = get_cluster_info(p_clusters, g_clusters)
        evaluator.update(cluster_info)

    score = {
        "coref_f": evaluator.get_f1(),
        "coref_p": evaluator.get_precision(),
        "coref_r": evaluator.get_recall(),
    }
    return score


def score_span_predictions(
    examples: Iterable[Example],
    *,
    output_prefix: str = DEFAULT_CLUSTER_PREFIX,
    **kwargs,
):
    """Evaluate reconstruction of the correct spans from gold heads."""
    scores = []
    for eg in examples:
        starts = []
        ends = []
        pred_starts = []
        pred_ends = []
        ref = eg.reference
        pred = eg.predicted
        for key, gold_sg in ref.spans.items():
            if len(gold_sg) == 0:
                # if there are no spans there's nothing to predict
                continue
            if not matches_coref_prefix(output_prefix, key):
                continue
            pred_sg = pred.spans[key]
            for gold_mention, pred_mention in zip(gold_sg, pred_sg):
                starts.append(gold_mention.start)
                ends.append(gold_mention.end)
                pred_starts.append(pred_mention.start)
                pred_ends.append(pred_mention.end)

        # it's possible there are no heads to predict from, in which case, skip
        if len(starts) == 0:
            continue

        # see how many are perfect
        cs = [a == b for a, b in zip(starts, pred_starts)]
        ce = [a == b for a, b in zip(ends, pred_ends)]
        correct = [int(a and b) for a, b in zip(cs, ce)]
        accuracy = sum(correct) / len(correct)

        scores.append(float(accuracy))
    out_key = f"span_{output_prefix}_accuracy"

    # it is possible there was nothing to score
    final = 0.0
    if len(scores) > 0:
        final = mean(scores)

    return {out_key: final}


# The following implementations of get_cluster_info(), get_markable_assignments,
# and ClusterEvaluator are adapted from coval, which is distributed under the
# MIT License.
# Copyright 2018 Nafise Sadat Moosavi
# See licenses/3rd_party_licenses.txt
def get_cluster_info(predicted_clusters, gold_clusters):
    p2g = get_markable_assignments(predicted_clusters, gold_clusters)
    g2p = get_markable_assignments(gold_clusters, predicted_clusters)
    # this is the data format used as input by the evaluator
    return (gold_clusters, predicted_clusters, g2p, p2g)


def get_markable_assignments(in_clusters, out_clusters):
    markable_cluster_ids = {}
    out_dic = {}
    for cluster_id, cluster in enumerate(out_clusters):
        for m in cluster:
            out_dic[m] = cluster_id

    for cluster in in_clusters:
        for im in cluster:
            for om in out_dic:
                if im == om:
                    markable_cluster_ids[im] = out_dic[om]
                    break

    return markable_cluster_ids


class ClusterEvaluator:
    def __init__(self, metric, beta=1, keep_aggregated_values=False):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta
        self.keep_aggregated_values = keep_aggregated_values

        if keep_aggregated_values:
            self.aggregated_p_num = []
            self.aggregated_p_den = []
            self.aggregated_r_num = []
            self.aggregated_r_den = []

    def update(self, coref_info):
        (
            key_clusters,
            sys_clusters,
            key_mention_sys_cluster,
            sys_mention_key_cluster,
        ) = coref_info

        pn, pd = self.metric(sys_clusters, key_clusters, sys_mention_key_cluster)
        rn, rd = self.metric(key_clusters, sys_clusters, key_mention_sys_cluster)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

        if self.keep_aggregated_values:
            self.aggregated_p_num.append(pn)
            self.aggregated_p_den.append(pd)
            self.aggregated_r_num.append(rn)
            self.aggregated_r_den.append(rd)

    def f1(self, p_num, p_den, r_num, r_den, beta=1):
        p = 0
        if p_den != 0:
            p = p_num / float(p_den)
        r = 0
        if r_den != 0:
            r = r_num / float(r_den)

        if p + r == 0:
            return 0

        return (1 + beta * beta) * p * r / (beta * beta * p + r)

    def get_f1(self):
        return self.f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        if self.r_num == 0:
            return 0

        return self.r_num / float(self.r_den)

    def get_precision(self):
        if self.p_num == 0:
            return 0

        return self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den

    def get_aggregated_values(self):
        return (
            self.aggregated_p_num,
            self.aggregated_p_den,
            self.aggregated_r_num,
            self.aggregated_r_den,
        )


def lea(input_clusters, output_clusters, mention_to_gold):
    num, den = 0, 0

    for c in input_clusters:
        if len(c) == 1:
            all_links = 1
            if (
                c[0] in mention_to_gold
                and len(output_clusters[mention_to_gold[c[0]]]) == 1
            ):
                common_links = 1
            else:
                common_links = 0
        else:
            common_links = 0
            all_links = len(c) * (len(c) - 1) / 2.0
            for i, m in enumerate(c):
                if m in mention_to_gold:
                    for m2 in c[i + 1 :]:
                        if (
                            m2 in mention_to_gold
                            and mention_to_gold[m] == mention_to_gold[m2]
                        ):
                            common_links += 1

        num += len(c) * common_links / float(all_links)
        den += len(c)

    return num, den


# This is coref related, but not from coval.
def doc2clusters(doc: Doc, prefix: str) -> List[List[Tuple[int, int]]]:
    """Given a doc, give the mention clusters.

    This is used for scoring.
    """
    out = []
    for name, val in doc.spans.items():
        if not matches_coref_prefix(prefix, name):
            continue

        cluster = []
        for mention in val:
            cluster.append((mention.start, mention.end))
        out.append(cluster)
    return out
