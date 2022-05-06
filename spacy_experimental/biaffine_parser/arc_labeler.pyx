# cython: infer_types=True, profile=True, binding=True

from itertools import islice
import numpy as np
from typing import Callable, Dict, Iterable, Optional
import spacy
from spacy import Language, Vocab
from spacy.errors import Errors
from spacy.pipeline.trainable_pipe cimport TrainablePipe
from spacy.tokens.token cimport Token
from spacy.tokens.doc cimport Doc
from spacy.training import Example, validate_get_examples, validate_examples
from spacy.util import minibatch
import srsly
from thinc.api import Config, Model, Ops, Optimizer
from thinc.api import to_numpy
from thinc.types import Floats2d, Ints1d, Tuple

from .eval import parser_score

default_model_config = """
[model]
@architectures = "spacy-experimental.Bilinear.v1"
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
    "experimental_arc_labeler",
    assigns=["token.dep"],
    default_config={
        "model": DEFAULT_ARC_LABELER_MODEL,
        "scorer": {"@scorers": "spacy-experimental.biaffine_parser_scorer.v1"}
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

    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, Floats2d]:
        validate_examples(examples, "ArcLabeler.get_loss")

        def loss_func(guesses, target, mask):
            d_scores = guesses - target
            d_scores *= mask
            loss = (d_scores ** 2).sum()
            return d_scores, loss

        target = np.zeros(scores.shape, dtype=scores.dtype)
        mask = np.zeros(scores.shape[0], dtype=scores.dtype)

        offset = 0
        for eg in examples:
            aligned_heads, aligned_labels = eg.get_aligned_parse(projectivize=False)
            for token in eg.predicted:
                gold_head = aligned_heads[token.i]
                gold_label = aligned_labels[token.i]

                # Do not learn from misaligned tokens, since we could no use
                # their correct head representations.
                if gold_head is not None and gold_label is not None:
                    target[offset, self._label_to_i[gold_label]] = 1.0
                    mask[offset] = 1.0

                offset += 1

        assert offset == target.shape[0]

        target = self.model.ops.asarray2f(target)
        mask = self.model.ops.asarray2f(np.expand_dims(mask, -1))

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

        doc_sample = []
        label_sample = []
        examples = list(islice(get_examples(), 10))
        for example in examples:
            doc_sample.append(example.predicted)
            gold_labels = example.get_aligned("DEP", as_string=True)
            gold_array = [[1.0 if tag == gold_tag else 0.0 for tag in self.labels] for gold_tag in gold_labels]
            label_sample.append(self.model.ops.asarray(gold_array, dtype="float32"))

        heads_sample = heads_gold(examples, self.model.ops)
        self.model.initialize(X=(doc_sample, heads_sample), Y=label_sample)

        # Store the input dimensionality. nI and nO are not stored explicitly
        # for PyTorch models. This makes it tricky to reconstruct the model
        # during deserialization. So, besides storing the labels, we also
        # store the number of inputs.
        bilinear = self.model.get_ref("bilinear")
        self.cfg["nI"] = bilinear.get_dim("nI")

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
        heads = heads_predicted(docs, self.model.ops)
        scores = self.model.predict((docs, heads))
        return to_numpy(scores.argmax(-1))

    def set_annotations(self, docs: Iterable[Doc], predictions):
        cdef Doc doc
        cdef Token token

        offset = 0
        for doc in docs:
            for sent in doc.sents:
                for token in sent:
                    label = self.cfg["labels"][predictions[offset]]
                    doc.c[token.i].dep = self.vocab.strings[label]
                    offset += 1

            for i in range(doc.length):
                if doc.c[i].head == 0:
                    doc.c[i].dep = self.vocab.strings['ROOT']

            # FIXME: we should enable this, but clears sentence boundaries
            # set_children_from_heads(doc.c, 0, doc.length)

        assert offset == predictions.shape[0]

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

        # TODO: train from the predicted heads instead - or at least make this an option
        gold_heads = heads_gold(examples, self.model.ops)

        scores, backprop_scores = self.model.begin_update((docs, gold_heads))
        loss, d_scores = self.get_loss(examples, scores)
        backprop_scores(d_scores)

        if sgd is not None:
            self.finish_update(sgd)
        losses[self.name] += loss

        return losses

    def add_label(self, label):
        """Add a new label to the pipe.

        label (str): The label to add.
        RETURNS (int): 0 if label is already present, otherwise 1.

        DOCS: https://spacy.io/api/tagger#add_label
        """
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

        self._initialize_from_disk()

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

        self._initialize_from_disk()

        model_deserializers = {
            "model": load_model,
        }
        spacy.util.from_disk(path, model_deserializers, exclude)

        return self

    def _initialize_from_disk(self):
        self._label_to_i = {label: i for i, label in enumerate(self.labels)}

        # The PyTorch model is constructed lazily, so we need to
        # explicitly initialize the model before deserialization.
        bilinear = self.model.get_ref("bilinear")
        if bilinear.has_dim("nI") is None:
            bilinear.set_dim("nI", self.cfg["nI"])
        if bilinear.has_dim("nO") is None:
            bilinear.set_dim("nO", len(self.labels))
        self.model.initialize()


def heads_gold(examples: Iterable[Example], ops: Ops) -> Ints1d:
    heads = []
    for eg in examples:
        aligned_heads, _ = eg.get_aligned_parse(projectivize=False)
        eg_offset = len(heads)
        for idx, head in enumerate(aligned_heads):
            if head is None:
                heads.append(eg_offset + idx)
            else:
                heads.append(eg_offset + head)

    return ops.asarray1i(heads)

def heads_predicted(docs: Iterable[Doc], ops: Ops) -> Ints1d:
    heads = []
    for doc in docs:
        doc_offset = len(heads)
        for idx, token in enumerate(doc):
            # FIXME: we should always get a head in prediction, make error?
            if token.head is None:
                heads.append(doc_offset + idx)
            else:
                heads.append(doc_offset + token.head.i)

    return ops.asarray1i(heads)
