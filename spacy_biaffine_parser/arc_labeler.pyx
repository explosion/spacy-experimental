# cython: infer_types=True, profile=True, binding=True

from itertools import islice
import numpy as np
from typing import Callable, Dict, Iterable, List, Optional
import spacy
from spacy import Language, Vocab
from spacy.errors import Errors
from spacy.pipeline.trainable_pipe cimport TrainablePipe
from spacy.scorer import Scorer
from spacy.symbols cimport dep, root
from spacy.tokens.token cimport Token
from spacy.tokens.doc cimport Doc, set_children_from_heads
from spacy.training import Example, validate_get_examples, validate_examples
from spacy.util import minibatch, registry
import srsly
from thinc.api import Config, Model, NumpyOps, Ops, Optimizer, Ragged, get_current_ops
from thinc.api import to_numpy
from thinc.types import Ints1d, Ragged, Tuple

from .eval import parser_score
from .mst import chu_liu_edmonds

def sents2lens(docs: List[Doc], *, ops: Optional[Ops] = None) -> Ints1d:
    if ops is None:
        ops = get_current_ops()

    lens = []
    for doc in docs:
        for sent in doc.sents:
            lens.append(sent.end - sent.start)

    return ops.asarray1i(lens)


default_model_config = """
[model]
@architectures = "BiaffineModel.v1"
hidden_width = 64

[model.tok2vec]
@architectures = "spacy.HashEmbedCNN.v2"
pretrained_vectors = null
width = 96
depth = 4
embed_size = 300
window_size = 1
maxout_pieces = 3
subword_features = true
"""
DEFAULT_ARC_LABELER_MODEL = Config().from_str(default_model_config)["model"]

@Language.factory(
    "arc_labeler",
    assigns=["token.dep"],
    default_config={
        "model": DEFAULT_ARC_LABELER_MODEL,
        "scorer": {"@scorers": "biaffine.parser_scorer.v1"}
    },
)
def make_arc_labeler(
    nlp: Language,
    name: str,
    model: Model,
    scorer: Optional[Callable],
):
    return ArcLabeler(nlp.vocab, model, name, scorer=scorer)


class ArcLabeler(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "arc_labeler",
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
        self._label_to_i = None

    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, float]:
        validate_examples(examples, "ArcLabeler.get_loss")

        def loss_func(guesses, target, mask):
            d_scores = guesses - target
            d_scores *= mask
            loss = (d_scores ** 2).sum()
            return d_scores, loss

        target = np.zeros_like(scores)
        mask = np.zeros(scores.shape[:2], dtype=scores.dtype)

        offset = 0
        for eg in examples:
            aligned_heads, aligned_labels = eg.get_aligned_parse(projectivize=False)
            for sent in eg.predicted.sents:
                for token in sent:
                    gold_head = aligned_heads[token.i]
                    if gold_head is not None:
                        if gold_head >= sent.start and gold_head < sent.end:
                            gold_head_idx = gold_head - sent.start

                            gold_label = aligned_labels[token.i]
                            if gold_label is not None:
                                target[offset, gold_head_idx, self._label_to_i[gold_label]] = 1.0
                                mask[offset, gold_head_idx] = 1.0

                    offset += 1

        assert offset == target.shape[0]

        target = self.model.ops.asarray4f(target)
        mask = self.model.ops.asarray4f(np.expand_dims(mask, -1))

        d_scores, loss = loss_func(scores, target, mask)

        return float(loss), d_scores

    def initialize(
        self, get_examples: Callable[[], Iterable[Example]], *, nlp: Language = None
    ):
        validate_get_examples(get_examples, "ArcLabeler.initialize")

        labels = set()
        for example in get_examples():
            for token in example.reference:
                if token.dep != 0:
                    labels.add(token.dep_)
        for label in sorted(labels):
            self.add_label(label)
        self._label_to_i = {label: i for i, label in enumerate(self.labels)}

        # nO can not be inferred on a tuplify layer in a chain.
        self.model.get_ref("biaffine").set_dim("nO", len(self.labels))

        doc_sample = []
        label_sample = []
        for example in islice(get_examples(), 10):
            # XXX: Should be example.x
            doc_sample.append(example.y)
            gold_labels = example.get_aligned("DEP", as_string=True)
            gold_array = [[1.0 if tag == gold_tag else 0.0 for tag in self.labels] for gold_tag in gold_labels]
            label_sample.append(self.model.ops.asarray(gold_array, dtype="float32"))
        span_sample = sents2lens(doc_sample, ops=self.model.ops)
        self.model.initialize(X=(doc_sample, span_sample), Y=(None, label_sample))

    @property
    def labels(self):
        return tuple(self.cfg["labels"])

    def pipe(self, docs, *, int batch_size=128):
        cdef Doc doc
        error_handler = self.get_error_handler()
        for batch in minibatch(docs, size=batch_size):
            batch_in_order = list(batch)
            try:
                by_length = sorted(batch, key=lambda doc: len(doc))
                for subbatch in minibatch(by_length, size=max(batch_size//4, 2)):
                    subbatch = list(subbatch)
                    predictions = self.predict(subbatch)
                    self.set_annotations(subbatch, predictions)
                yield from batch_in_order
            except Exception as e:
                error_handler(self.name, self, batch_in_order, e)


    def predict(self, docs: Iterable[Doc]):
        docs = list(docs)
        lens = sents2lens(docs, ops=self.model.ops)
        scores = self.model.predict((docs, lens))
        return lens, scores

    def set_annotations(self, docs: Iterable[Doc], lens_scores):
        cdef Doc doc
        cdef Token token

        # XXX: predict best in `predict`

        lens, scores = lens_scores
        scores = to_numpy(scores.argmax(-1))
        lens = to_numpy(lens)

        offset = 0
        for doc in docs:
            for sent in doc.sents:
                for token in sent:
                    head_i = token.i + doc.c[token.i].head
                    head = head_i - sent.start
                    label = self.cfg["labels"][scores[offset, head]]
                    doc.c[token.i].dep = self.vocab.strings[label]
                    offset += 1

                lens = lens[1:]


            for i in range(doc.length):
                if doc.c[i].head == 0:
                    doc.c[i].dep = self.vocab.strings['ROOT']

            # XXX: we should enable this, but clears sentence boundaries
            # set_children_from_heads(doc.c, 0, doc.length)

        assert offset == scores.shape[0]
        assert len(lens) == 0

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
        validate_examples(examples, "ArcLabeler.update")

        if not any(len(eg.predicted) if eg.predicted else 0 for eg in examples):
            # Handle cases where there are no tokens in any docs.
            return losses

        docs = [eg.predicted for eg in examples]

        lens = sents2lens(docs, ops=self.model.ops)
        if lens.sum() == 0:
            return losses

        # set_dropout_rate(self.model, drop)
        scores, backprop_scores = self.model.begin_update((docs, lens))
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

    def from_bytes(self, bytes_data, *, exclude=tuple()):
        deserializers = {
            "cfg": lambda b: self.cfg.update(srsly.json_loads(b)),
            "vocab": lambda b: self.vocab.from_bytes(b, exclude=exclude),
        }
        spacy.util.from_bytes(bytes_data, deserializers, exclude)

        self._initialize_from_labels()

        model_deserializers = {
            "model": lambda b: self.model.from_bytes(b),
        }
        spacy.util.from_bytes(bytes_data, model_deserializers, exclude)

        return self

    def to_bytes(self, *, exclude=tuple()):
        serializers = {
            "cfg": lambda: srsly.json_dumps(self.cfg),
            "model": lambda: self.model.to_bytes(),
            "vocab": lambda: self.vocab.to_bytes(exclude=exclude),
        }

        return spacy.util.to_bytes(serializers, exclude)

    def to_disk(self, path, exclude=tuple()):
        path = spacy.util.ensure_path(path)
        serializers = {
            "cfg": lambda p: srsly.write_json(p, self.cfg),
            "model": lambda p: self.model.to_disk(p),
            "vocab": lambda p: self.vocab.to_disk(p, exclude=exclude),
        }
        spacy.util.to_disk(path, serializers, exclude)

    def from_disk(self, path, exclude=tuple()):
        def load_model(p):
            try:
                with open(p, "rb") as mfile:
                    self.model.from_bytes(mfile.read())
            except AttributeError:
                raise ValueError(Errors.E149) from None

        deserializers = {
            "cfg": lambda p: self.cfg.update(srsly.read_json(p)),
            "vocab": lambda p: self.vocab.from_disk(p, exclude=exclude),
        }
        spacy.util.from_disk(path, deserializers, exclude)

        self._initialize_from_labels()

        model_deserializers = {
            "model": load_model,
        }
        spacy.util.from_disk(path, model_deserializers, exclude)

        return self

    def _initialize_from_labels(self):
        self._label_to_i = {label: i for i, label in enumerate(self.labels)}

        # The PyTorch model is constructed lazily, so we need to
        # explicitly initialize the model before deserialization.
        label_model = self.model.get_ref("biaffine")
        if label_model.has_dim("nO") is None:
            label_model.set_dim("nO", len(self.labels))
        self.model.initialize()


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
