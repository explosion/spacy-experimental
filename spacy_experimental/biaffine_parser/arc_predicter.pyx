# cython: infer_types=True, binding=True

from typing import Callable, Dict, Iterable, List, Optional
from itertools import islice
from collections import deque
import numpy as np
import spacy
from spacy import Language, Vocab
from spacy.errors import Errors
from spacy.pipeline.dep_parser import parser_score
from spacy.pipeline.trainable_pipe cimport TrainablePipe
from spacy.tokens.token cimport Token
from spacy.tokens.doc cimport Doc
from spacy.training import Example, validate_get_examples, validate_examples
from spacy.util import minibatch
import srsly
from thinc.api import Config, Model, NumpyOps, Ops, Optimizer
from thinc.api import to_numpy
from thinc.types import Floats1d, Floats2d, Ints1d, Tuple

from .mst import mst_decode
from ._util import lengths2offsets


NUMPY_OPS = NumpyOps()


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
        "overwrite": False,
        "scorer": {"@scorers": "spacy.parser_scorer.v1"},
        "senter_name": None,
        "max_tokens": 100,
    },
)
def make_arc_predicter(
    nlp: Language,
    name: str,
    model: Model,
    overwrite: bool,
    scorer: Optional[Callable],
    senter_name: Optional[str],
    max_tokens: int,
):
    return ArcPredicter(nlp.vocab, model, name, max_tokens=max_tokens, overwrite=overwrite, scorer=scorer, senter_name=senter_name)


class ArcPredicter(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "arc_predicter",
        *,
        max_tokens: int,
        overwrite: bool,
        scorer: Callable,
        senter_name: Optional[str] = None,
    ):
        self.name = name
        self.model = model
        self.max_tokens = max_tokens
        self.senter_name = senter_name
        self.vocab = vocab
        cfg = {"labels": [], "overwrite": overwrite}
        self.cfg = dict(sorted(cfg.items()))
        self.scorer = scorer

    def get_loss(self, examples: Iterable[Example], scores: List[Floats2d],
                 docs_split_lengths: List[Ints1d]) -> Tuple[float, Floats2d]:
        validate_examples(examples, "ArcPredicter.get_loss")

        def loss_func(guesses, target, mask):
            d_scores = guesses - target
            d_scores *= mask
            loss = (d_scores ** 2).sum()
            return d_scores, loss

        # We want to compute all the losses at once to avoid too many kernel runs.
        scores = self.model.ops.flatten([split_scores.reshape(-1) for split_scores in scores])

        target = np.zeros(scores.shape, dtype=scores.dtype)
        mask = np.zeros(scores.shape, dtype=scores.dtype)

        offset = 0
        for eg, doc_split_lengths in zip(examples, docs_split_lengths):
            aligned_heads, _ = eg.get_aligned_parse(projectivize=False)
            sent_start = 0
            split_offsets = lengths2offsets(doc_split_lengths)
            for split_offset, split_length in zip(split_offsets, doc_split_lengths):
                for i in range(split_length):
                    gold_head = aligned_heads[split_offset + i]
                    if gold_head is not None:
                        # We only use the loss for token for which the correct head
                        # lies within the sentence boundaries.
                        if split_offset <= gold_head < split_offset + split_length:
                            gold_head_idx = gold_head - split_offset
                            target[offset + gold_head_idx] = 1.0
                            mask[offset:offset + split_length] = 1
                    offset += split_length

                sent_start += split_length

        assert offset == target.shape[0]

        target = self.model.ops.asarray1f(target)
        mask = self.model.ops.asarray1f(mask)

        d_scores, loss = loss_func(scores, target, mask)


        split_lengths = [split_length for doc_split_lengths in docs_split_lengths for split_length in doc_split_lengths]
        return float(loss), unflatten_matrix(d_scores, split_lengths)

    def initialize(
        self, get_examples: Callable[[], Iterable[Example]], *, nlp: Language = None
    ):
        validate_get_examples(get_examples, "ArcPredicter.initialize")

        doc_sample = []
        for example in islice(get_examples(), 10):
            doc_sample.append(example.predicted)

        # For initialization, we don't need correct sentence boundaries.
        lengths_sample = [NUMPY_OPS.asarray1i([len(doc)]) for doc in doc_sample]
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

        if self.senter_name:
            docs_split_lengths = split_lazily(docs, ops=self.model.ops, max_tokens=self.max_tokens, senter_name=self.senter_name)
        else:
            docs_split_lengths = sents2lens(docs, ops=self.model.ops)

        scores = self.model.predict((docs, docs_split_lengths))

        # Flatten once when on GPU and do one d2h transfer.
        scores = self.model.ops.flatten([split_scores.reshape(-1) for split_scores in scores])
        scores = to_numpy(scores)

        heads = []
        for doc_split_lengths in docs_split_lengths:
            # Flat score array to per-split score matrices.
            scores_len = (doc_split_lengths ** 2).sum()
            scores_matrix = unflatten_matrix(scores[:scores_len], doc_split_lengths)

            # Decode split score matrices.
            doc_heads = []
            for split_scores in scores_matrix:
                split_heads = mst_decode(split_scores)
                split_heads = [head - i for (i, head) in enumerate(split_heads)]
                doc_heads.extend(split_heads)

            scores = scores[scores_len:]

            heads.append(doc_heads)

        assert len(scores) == 0

        return heads

    def set_annotations(self, docs: Iterable[Doc], heads):
        cdef Doc doc
        cdef Token token

        for (doc, doc_heads) in zip(docs, heads):
            for token, head in zip(doc, doc_heads):
                doc.c[token.i].head = head
                # FIXME: Set the dependency relation to a stub, so that
                # we can evaluate UAS.
                doc.c[token.i].dep = self.vocab.strings['dep']

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

        if self.senter_name:
            lens = split_lazily(docs, ops=self.model.ops, max_tokens=self.max_tokens, senter_name=self.senter_name)
        else:
            lens = sents2lens(docs, ops=self.model.ops)
        if sum([sum(doc_lens) for doc_lens in lens]) == 0:
            return losses

        scores, backprop_scores = self.model.begin_update((docs, lens))
        loss, d_scores = self.get_loss(examples, scores, lens)
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


def sents2lens(docs: List[Doc], *, ops: Ops) -> List[Ints1d]:
    """Get the lengths of sentences."""
    lens = []
    for doc in docs:
        doc_lens = []
        for sent in doc.sents:
            doc_lens.append(sent.end - sent.start)
        lens.append(NUMPY_OPS.asarray1i(doc_lens))
    return lens


def split_lazily(docs: List[Doc], *, ops: Ops, max_tokens: int, senter_name: str) -> List[Ints1d]:
    lens = []
    for doc in docs:
        activations = doc.activations.get(senter_name, None)
        if activations is None:
            raise ValueError(f"Lazy splitting requires senter pipe `{senter_name}` to have ",
                              "`save_activations` enabled.\nDuring training, `senter` must be "
                              "in the list of annotating components.")
        scores = activations['probabilities']
        doc_lens = _split_lazily_doc(ops, scores[:, 1], max_tokens)
        lens.append(NUMPY_OPS.asarray1i(doc_lens))

    assert sum([sum(split_lens) for split_lens in lens]) == sum([len(doc) for doc in docs])

    return lens


def _split_lazily_doc(ops: Ops, scores: Floats2d, max_tokens: int) -> List[int]:
    lens = []
    stack = deque([scores])
    while stack:
        scores = stack.pop()
        if len(scores) <= max_tokens:
            lens.append(len(scores))
        else:
            # Find the best splitting point. Exclude the first token, because it
            # wouldn't split the current partition (leading to infinite recursion).
            start = ops.xp.argmax(scores[1:]) + 1
            # Initial split goes last, so that it is taken off the stack first
            # in the next iteration.
            stack.append(scores[start:])
            stack.append(scores[:start])
    return lens


def unflatten_matrix(scores: Floats1d, lengths: Iterable[int]) -> List[Floats2d]:
    scores_matrix = []
    for length in lengths:
        scores_matrix.append(scores[:length * length].reshape((length, length)))
        scores = scores[length * length:]
    assert scores.size == 0
    return scores_matrix
