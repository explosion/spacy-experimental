# cython: infer_types=True, profile=True, binding=True

from itertools import islice
import numpy as np
from typing import Callable, Dict, Iterable, List, Optional
from spacy import Language, Vocab
from spacy.errors import Errors
from spacy.ml import extract_spans
from spacy.pipeline.trainable_pipe cimport TrainablePipe
from spacy.scorer import Scorer
from spacy.symbols cimport dep, root
from spacy.tokens.token cimport Token
from spacy.tokens.doc cimport Doc, set_children_from_heads
from spacy.training import Example, validate_get_examples, validate_examples
from spacy.util import registry
from thinc.api import Model, NumpyOps, Ops, Optimizer, Ragged, get_current_ops
from thinc.api import to_numpy
from thinc.types import Ints1d, Ragged, Tuple

from .eval import score_deps
from .mst import chu_liu_edmonds

def sents2lens(docs: List[Doc], *, ops: Optional[Ops] = None) -> Ints1d:
    if ops is None:
        ops = get_current_ops()

    lengths = []
    for doc in docs:
        for sent in doc.sents:
            lengths.append(sent.end - sent.start)

    return ops.asarray1i(lengths)




def parser_score(examples, **kwargs):
    return score_deps(examples)


@registry.scorers("biaffine.parser_scorer.v1")
def make_parser_scorer():
    return parser_score


@Language.factory(
    "biaffine_parser",
    default_config={"scorer": {"@scorers": "biaffine.parser_scorer.v1"}},
)
def make_biaffine_parser(
    nlp: Language,
    name: str,
    model: Model,
    scorer: Optional[Callable],
):
    return BiaffineParser(nlp.vocab, model, name, scorer=scorer)


cdef class BiaffineParser(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "biaffine_parser",
        *,
        overwrite=False,
        scorer=parser_score
    ):
        self.name = name
        self.model = model
        self.vocab = vocab
        cfg = {"labels": [], "overwrite": overwrite}
        self.cfg = dict(sorted(cfg.items()))
        self.scorer = scorer

    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, float]:
        validate_examples(examples, "BiaffineParser.get_loss")

        def loss_func(guesses, target, mask):
            d_scores = guesses - target
            d_scores *= mask.reshape(d_scores.shape[0], 1)
            loss = (d_scores ** 2).sum()
            return d_scores, loss

        target = np.zeros_like(scores)
        mask = np.zeros(scores.shape[0], dtype=scores.dtype)

        offset = 0
        for eg in examples:
            aligned_heads, _ = eg.get_aligned_parse(projectivize=False)
            for sent in eg.predicted.sents:
                for token in sent:
                    head = aligned_heads[token.i]
                    if head is not None and head >= sent.start and head < sent.end:
                        mask[offset] = 1
                        target[offset, head - sent.start] = 1.0

                    offset += 1

        assert offset == target.shape[0]

        target = self.model.ops.asarray_f(target)
        mask = self.model.ops.asarray_f(mask)

        d_scores, loss = loss_func(scores, target, mask)

        return float(loss), d_scores

    def initialize(
        self, get_examples: Callable[[], Iterable[Example]], *, nlp: Language = None
    ):
        validate_get_examples(get_examples, "BiaffineParser.initialize")

        labels = set()
        for example in get_examples():
            for token in example.reference:
                if token.dep != 0:
                    labels.add(token.dep_)
        for label in sorted(labels):
            self.add_label(label)

        doc_sample = []
        label_sample = []
        for example in islice(get_examples(), 10):
            # XXX: Should be example.x
            doc_sample.append(example.y)
            gold_labels = example.get_aligned("DEP", as_string=True)
            gold_array = [[1.0 if tag == gold_tag else 0.0 for tag in self.labels] for gold_tag in gold_labels]
            label_sample.append(self.model.ops.asarray(gold_array, dtype="float32"))
        span_sample = sents2lens(doc_sample, ops=self.model.ops)
        self.model.initialize(X=(doc_sample, span_sample), Y=label_sample)

    @property
    def labels(self):
        return tuple(self.cfg["labels"])

    def predict(self, docs: Iterable[Doc]):
        docs = list(docs)
        lengths = sents2lens(docs, ops=self.model.ops)
        scores = self.model.predict((docs, lengths))
        return lengths, scores

    def set_annotations(self, docs: Iterable[Doc], spans_scores):
        cdef Doc doc
        cdef Token token

        # XXX: predict best in `predict`

        lengths, scores = spans_scores
        lengths = to_numpy(lengths)
        scores = to_numpy(scores)

        sent_offset = 0
        predicted_heads = scores.argmax(-1)
        for doc in docs:
            for sent in doc.sents:
                heads = mst_decode(scores[sent_offset:sent_offset + lengths[0], :lengths[0]])
                for i, head in enumerate(heads):
                    dep_i = sent.start + i
                    head_i = sent.start + head
                    doc.c[dep_i].head = head_i - dep_i
                    doc.c[dep_i].dep = self.vocab.strings['dep']

                sent_offset += lengths[0]
                lengths = lengths[1:]


            for i in range(doc.length):
                if doc.c[i].head == 0:
                    doc.c[i].dep = self.vocab.strings['ROOT']

            # XXX: we should enable this, but clears sentence boundaries
            # set_children_from_heads(doc.c, 0, doc.length)

        assert len(lengths) == 0
        assert sent_offset == predicted_heads.shape[0]

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        validate_examples(examples, "BiaffineParser.update")

        if not any(len(eg.predicted) if eg.predicted else 0 for eg in examples):
            # Handle cases where there are no tokens in any docs.
            return losses

        docs = [eg.predicted for eg in examples]

        spans = sents2lens(docs, ops=self.model.ops)
        if spans.sum() == 0:
            return losses

        # set_dropout_rate(self.model, drop)
        scores, backprop_scores = self.model.begin_update((docs, spans))
        loss, d_scores = self.get_loss(examples, scores)
        backprop_scores(d_scores)

        if sgd is not None:
            self.finish_update(sgd)
        losses[self.name] += loss

        return losses

    def add_label(self, label):
        if not isinstance(label, str):
            raise ValueError(Errors.E187)
        if label in self.labels:
            return 0
        self.cfg["labels"].append(label)
        self.vocab.strings.add(label)
        return 1

def mst_decode(sent_scores):
    """Apply MST decoding"""

    # Within spacy, a root is encoded as a token that attaches to itself
    # (relative offset 0). However, the decoder uses a specific vertex,
    # typically 0. So, we stub an additional root vertex to accomodate
    # this.

    # We expect a biaffine attention matrix.
    assert sent_scores.shape[0] == sent_scores.shape[1]

    seq_len = sent_scores.shape[0]

    # Create score matrix with root row/column.
    with_root = np.full((seq_len+1, seq_len+1), -10000, dtype=sent_scores.dtype)
    with_root[1:,1:] = sent_scores

    with_root[1:,0] = sent_scores.diagonal()
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
