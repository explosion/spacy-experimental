from typing import Iterable

from spacy.scorer import PRFScore, Scorer
from spacy.training import Example


def parser_score(examples, **kwargs):
    return score_deps(examples)


def make_parser_scorer():
    return parser_score


# Warning: use of this should be replaced by spaCy's score_deps. This
# is just for development, no proper evaluation.
def score_deps(examples: Iterable[Example]):
    """Dependency scoring function that takes into account incorrect
    boundaries."""
    labelled = PRFScore()
    unlabelled = PRFScore()

    for example in examples:
        gold_deps = set()
        pred_deps = set()
        aligned_heads, aligned_labels = example.get_aligned_parse(projectivize=False)

        for sent in example.predicted.sents:
            for token in sent:
                gold_head = aligned_heads[token.i]
                gold_label = aligned_labels[token.i]
                if gold_head is None:
                    continue
                if gold_head < sent.start or gold_head >= sent.end:
                    # We can never correctly predict heads when the sentence
                    # boundary predictor placed the gold head out of the sentence.
                    continue
                gold_deps.add((token.i, gold_head, gold_label))
                pred_deps.add((token.i, token.head.i, token.dep_))

        labelled.score_set(pred_deps, gold_deps)
        unlabelled.score_set(
            {dep[:2] for dep in pred_deps}, {dep[:2] for dep in gold_deps}
        )

    bound_scores = {
        "bound_dep_las": labelled.fscore,
        "bound_dep_uas": unlabelled.fscore,
    }

    no_bound_scores = Scorer.score_deps(examples, "dep")

    return {**bound_scores, **no_bound_scores}
