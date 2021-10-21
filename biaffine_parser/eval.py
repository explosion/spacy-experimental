from typing import Iterable
from spacy.scorer import PRFScore
from spacy.training import Example

def score_deps(examples: Iterable[Example]):
    """Dependency scoring function that takes into account incorrect
    boundaries."""
    unlabelled = PRFScore()
    offset = 0
    gold_deps = set()
    pred_deps = set()
    for example in examples:
        aligned_gold, _ = example.get_aligned_parse(projectivize=False)
        for sent in example.predicted.sents:
            for token in sent:
                gold_head = aligned_gold[token.i]
                if gold_head == None:
                    continue
                if gold_head < sent.start or gold_head >= sent.end:
                    # We can never correctly predict heads when the sentence
                    # boundary predictor placed the gold head out of the sentence.
                    continue
                gold_deps.add((token.i, gold_head))
                pred_deps.add((token.i, token.head.i))
    unlabelled.score_set(pred_deps, gold_deps)

    return {
        "dep_uas": unlabelled.fscore,
    }
