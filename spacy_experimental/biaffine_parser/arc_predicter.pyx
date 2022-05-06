# cython: infer_types=True, profile=True, binding=True

from itertools import islice
import numpy as np
from typing import Callable, Dict, Iterable, List, Optional
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
from .mst import mst_decode


default_model_config = """
[model]
@architectures = "spacy-experimental.PairwiseBilinear.v1"
hidden_width = 64
nO = 1

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
DEFAULT_ARC_PREDICTER_MODEL = Config().from_str(default_model_config)["model"]

@Language.factory(
    "experimental_arc_predicter",
    assigns=["token.head"],
    default_config={
        "model": DEFAULT_ARC_PREDICTER_MODEL,
        "scorer": {"@scorers": "spacy-experimental.biaffine_parser_scorer.v1"}
    },
)
def make_arc_predicter(
    nlp: Language,
    name: str,
    model: Model,
    scorer: Optional[Callable],
):
    return ArcPredicter(nlp.vocab, model, name, scorer=scorer)


class ArcPredicter(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "arc_predicter",
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

    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, Floats2d]:
        validate_examples(examples, "ArcPredicter.get_loss")

        def loss_func(guesses, target, mask):
            d_scores = guesses - target
            d_scores *= mask
            loss = (d_scores ** 2).sum()
            return d_scores, loss

        target = np.zeros(scores.shape, dtype=scores.dtype)
        mask = np.zeros(scores.shape[0], dtype=scores.dtype)

        offset = 0
        for eg in examples:
            aligned_heads, _ = eg.get_aligned_parse(projectivize=False)
            # TODO: what if eg.predicted doesn't contain sents? (raise error?)
            for sent in eg.predicted.sents:
                for token in sent:
                    gold_head = aligned_heads[token.i]
                    if gold_head is not None:
                        # We only use the loss for token for which the correct head
                        # lies within the sentence boundaries.
                        if sent.start <= gold_head < sent.end:
                            gold_head_idx = gold_head - sent.start
                            target[offset, gold_head_idx] = 1.0
                            mask[offset] = 1

                    offset += 1

        assert offset == target.shape[0]

        target = self.model.ops.asarray2f(target)
        mask = self.model.ops.asarray2f(np.expand_dims(mask, -1))

        d_scores, loss = loss_func(scores, target, mask)

        return float(loss), d_scores

    def initialize(
        self, get_examples: Callable[[], Iterable[Example]], *, nlp: Language = None
    ):
        validate_get_examples(get_examples, "ArcPredicter.initialize")

        doc_sample = []
        for example in islice(get_examples(), 10):
            doc_sample.append(example.predicted)

        # For initialization, we don't need correct sentence boundaries.
        lengths_sample = self.model.ops.asarray1i([len(doc) for doc in doc_sample])
        self.model.initialize(X=(doc_sample, lengths_sample))

        # Store the input dimensionality. nI and nO are not stored explicitly
        # for PyTorch models. This makes it tricky to reconstruct the model
        # during deserialization. So, besides storing the labels, we also
        # store the number of inputs.
        pairwise_bilinear = self.model.get_ref("pairwise_bilinear")
        self.cfg["nI"] = pairwise_bilinear.get_dim("nI")

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

        lens = to_numpy(lens)
        scores = to_numpy(scores)

        heads = []
        sent_offset = 0
        for doc in docs:
            doc_heads = []
            for sent in doc.sents:
                sent_heads = mst_decode(scores[sent_offset:sent_offset + lens[0], :lens[0]])
                doc_heads.append(sent_heads)

                sent_offset += lens[0]
                lens = lens[1:]

            heads.append(doc_heads)

        assert len(lens) == 0
        assert sent_offset == scores.shape[0]

        return heads

    def set_annotations(self, docs: Iterable[Doc], heads):
        cdef Doc doc
        cdef Token token

        for (doc, doc_heads) in zip(docs, heads):
            for (sent, sent_heads) in zip(doc.sents, doc_heads):
                for i, head in enumerate(sent_heads):
                    dep_i = sent.start + i
                    head_i = sent.start + head
                    doc.c[dep_i].head = head_i - dep_i
                    # FIXME: Set the dependency relation to a stub, so that
                    # we can evaluate UAS.
                    doc.c[dep_i].dep = self.vocab.strings['dep']

            for i in range(doc.length):
                if doc.c[i].head == 0:
                    doc.c[i].dep = self.vocab.strings['ROOT']

            # FIXME: we should enable this, but clears sentence boundaries
            # set_children_from_heads(doc.c, 0, doc.length)

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
        validate_examples(examples, "ArcPredicter.update")

        if not any(len(eg.predicted) if eg.predicted else 0 for eg in examples):
            # Handle cases where there are no tokens in any docs.
            return losses

        docs = [eg.predicted for eg in examples]

        lens = sents2lens(docs, ops=self.model.ops)
        if lens.sum() == 0:
            return losses

        scores, backprop_scores = self.model.begin_update((docs, lens))
        loss, d_scores = self.get_loss(examples, scores)
        backprop_scores(d_scores)

        if sgd is not None:
            self.finish_update(sgd)
        losses[self.name] += loss

        return losses

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
        # We are lazily initializing the PyTorch model. If a PyTorch transformer
        # is used, which is also lazily initialized, then the model did not have
        # the chance yet to get its input shape.
        pairwise_bilinear = self.model.get_ref("pairwise_bilinear")
        if pairwise_bilinear.has_dim("nI") is None:
            pairwise_bilinear.set_dim("nI", self.cfg["nI"])

        self.model.initialize()

def sents2lens(docs: List[Doc], *, ops: Ops) -> Ints1d:
    """Get the lengths of sentences."""
    lens = []
    for doc in docs:
        for sent in doc.sents:
            lens.append(sent.end - sent.start)

    return ops.asarray1i(lens)
