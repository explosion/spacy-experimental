from spacy.scorer import Scorer
from spacy.util import registry


def tokenizer_score(examples, **kwargs):
    return Scorer.score_tokenization(examples)


def make_tokenizer_scorer():
    return tokenizer_score


def tokenizer_senter_score(examples, **kwargs):
    def has_sents(doc):
        return doc.has_annotation("SENT_START")

    results = Scorer.score_tokenization(examples)
    results.update(Scorer.score_spans(examples, "sents", has_annotation=has_sents, **kwargs))
    del results["sents_per_type"]
    return results


def make_tokenizer_senter_scorer():
    return tokenizer_senter_score
