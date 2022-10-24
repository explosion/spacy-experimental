from typing import Iterable, Optional, Dict, Callable, Any, List
from functools import partial
import warnings

from thinc.types import Floats2d, Floats3d
from thinc.api import Model, Config, Optimizer
from thinc.api import set_dropout_rate, to_categorical
from itertools import islice
import srsly

from spacy.pipeline import TrainablePipe
from spacy.language import Language
from spacy.training import Example, validate_examples, validate_get_examples
from spacy.errors import Errors
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.util import from_bytes, from_disk

from .coref_util import MentionClusters
from .coref_util import DEFAULT_CLUSTER_PREFIX, DEFAULT_CLUSTER_HEAD_PREFIX

from .coref_scorer import doc2clusters, score_span_predictions, matches_coref_prefix

default_span_resolver_config = """
[model]
@architectures = "spacy-experimental.SpanResolver.v1"
hidden_size = 1024
distance_embedding_size = 64
conv_channels = 4
window_size = 1
max_distance = 128
prefix = "coref_head_clusters"

[model.tok2vec]
@architectures = "spacy.Tok2Vec.v2"

[model.tok2vec.embed]
@architectures = "spacy.MultiHashEmbed.v1"
width = 64
attrs = ["NORM", "LOWER", "PREFIX", "SUFFIX", "SHAPE"]
rows = [5000, 2500, 1000, 2500, 2500]
include_static_vectors = false

[model.tok2vec.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = ${model.tok2vec.embed.width}
window_size = 1
maxout_pieces = 3
depth = 2
"""
DEFAULT_SPAN_RESOLVER_MODEL = Config().from_str(default_span_resolver_config)["model"]


def span_resolver_scorer(
    examples: Iterable[Example], input_prefix: str, output_prefix: str, **kwargs
) -> Dict[str, Any]:
    return score_span_predictions(
        examples, input_prefix=input_prefix, output_prefix=output_prefix, **kwargs
    )


def make_span_resolver_scorer(
    input_prefix: str = DEFAULT_CLUSTER_HEAD_PREFIX,
    output_prefix: str = DEFAULT_CLUSTER_PREFIX,
):
    return partial(
        span_resolver_scorer, input_prefix=input_prefix, output_prefix=output_prefix
    )


@Language.factory(
    "experimental_span_resolver",
    assigns=["doc.spans"],
    requires=["doc.spans"],
    default_config={
        "model": DEFAULT_SPAN_RESOLVER_MODEL,
        "input_prefix": DEFAULT_CLUSTER_HEAD_PREFIX,
        "output_prefix": DEFAULT_CLUSTER_PREFIX,
        "scorer": {
            "@scorers": "spacy-experimental.span_resolver_scorer.v1",
            "output_prefix": DEFAULT_CLUSTER_PREFIX,
        },
    },
    default_score_weights={"span_accuracy": 1.0},
)
def make_span_resolver(
    nlp: Language,
    name: str,
    model: Model[List[Doc], Floats2d],
    input_prefix: str,
    output_prefix: str,
    scorer: Optional[Callable],
) -> "SpanResolver":
    """Create a SpanResolver component."""
    return SpanResolver(
        nlp.vocab,
        model,
        name,
        input_prefix=input_prefix,
        output_prefix=output_prefix,
        scorer=scorer,
    )


class SpanResolver(TrainablePipe):
    """Pipeline component to resolve one-token spans to full spans.

    Used in coreference resolution.

    DOCS: https://spacy.io/api/span_resolver
    """

    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "span_resolver",
        *,
        input_prefix: str = DEFAULT_CLUSTER_HEAD_PREFIX,
        output_prefix: str = DEFAULT_CLUSTER_PREFIX,
        scorer: Optional[Callable] = span_resolver_scorer,
    ) -> None:
        self.vocab = vocab
        self.model = model
        self.name = name
        self.input_prefix = input_prefix
        self.output_prefix = output_prefix

        self.scorer = scorer
        self.cfg: Dict[str, Any] = {}

    def predict(self, docs: Iterable[Doc]) -> List[MentionClusters]:
        """Apply the pipeline's model to a batch of docs, without modifying them.
        Return the list of predicted span clusters.

        docs (Iterable[Doc]): The documents to predict.
        RETURNS (List[MentionClusters]): The model's prediction for each document.

        DOCS: https://spacy.io/api/span_resolver#predict
        """
        # for now pretend there's just one doc

        out = []
        for doc in docs:
            span_scores = self.model.predict([doc])
            if span_scores.size:
                # the information about clustering has to come from the input docs
                # first let's convert the scores to a list of span idxs
                start_scores = span_scores[:, :, 0]
                end_scores = span_scores[:, :, 1]
                starts = start_scores.argmax(axis=1)
                ends = end_scores.argmax(axis=1)

                # get the old clusters (shape will be preserved)
                clusters = doc2clusters(doc, self.input_prefix)
                cidx = 0
                out_clusters = []
                for cluster in clusters:
                    ncluster = []
                    for mention in cluster:
                        ncluster.append((starts[cidx], ends[cidx]))
                        cidx += 1
                    out_clusters.append(ncluster)
            else:
                out_clusters = []
            out.append(out_clusters)
        return out

    def set_annotations(self, docs: Iterable[Doc], clusters_by_doc) -> None:
        """Modify a batch of Doc objects, using pre-computed scores.

        docs (Iterable[Doc]): The documents to modify.
        clusters: The span clusters, produced by SpanResolver.predict.

        DOCS: https://spacy.io/api/span_resolver#set_annotations
        """
        for doc, clusters in zip(docs, clusters_by_doc):
            for ii, cluster in enumerate(clusters, 1):
                # Note the +1, since model end indices are inclusive
                spans = [doc[int(mm[0]) : int(mm[1]) + 1] for mm in cluster]
                doc.spans[f"{self.output_prefix}_{ii}"] = spans

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Learn from a batch of documents and gold-standard information,
        updating the pipe's model. Delegates to predict and get_loss.

        examples (Iterable[Example]): A batch of Example objects.
        drop (float): The dropout rate.
        sgd (thinc.api.Optimizer): The optimizer.
        losses (Dict[str, float]): Optional record of the loss during training.
            Updated using the component name as the key.
        RETURNS (Dict[str, float]): The updated losses dictionary.

        DOCS: https://spacy.io/api/span_resolver#update
        """
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        validate_examples(examples, "SpanResolver.update")
        if not any(len(eg.reference) if eg.reference else 0 for eg in examples):
            # Handle cases where there are no tokens in any docs.
            return losses
        set_dropout_rate(self.model, drop)

        total_loss = 0
        for eg in examples:
            if eg.x.text != eg.y.text:
                # TODO assign error number
                raise ValueError(
                    """Text, including whitespace, must match between reference and
                    predicted docs in span resolver training.
                    """
                )
            span_scores, backprop = self.model.begin_update([eg.predicted])

            if span_scores.size == 0:
                # This can happen if there are no input clusters.
                continue
            loss, d_scores = self.get_loss([eg], span_scores)
            total_loss += loss
            backprop((d_scores))

        if sgd is not None:
            self.finish_update(sgd)
        losses[self.name] += total_loss
        return losses

    def rehearse(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        # TODO this should be added later
        raise NotImplementedError(
            Errors.E931.format(
                parent="SpanResolver", method="add_label", name=self.name
            )
        )

    def add_label(self, label: str) -> int:
        """Technically this method should be implemented from TrainablePipe,
        but it is not relevant for this component.
        """
        raise NotImplementedError(
            Errors.E931.format(
                parent="SpanResolver", method="add_label", name=self.name
            )
        )

    def get_loss(
        self,
        examples: Iterable[Example],
        span_scores: Floats3d,
    ):
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores.

        examples (Iterable[Examples]): The batch of examples.
        scores: Scores representing the model's predictions.
        RETURNS (Tuple[float, float]): The loss and the gradient.

        DOCS: https://spacy.io/api/span_resolver#get_loss
        """
        ops = self.model.ops

        # NOTE This is doing fake batching, and should always get a list of one example
        assert len(list(examples)) == 1, "Only fake batching is supported."

        # NOTE Within this component, end token indices are *inclusive*. This
        # is different than normal Python/spaCy representations, but has the
        # advantage that the set of possible start and end indices is the same.
        for eg in examples:
            # starts and ends are gold starts and ends (Ints1d)
            starts = []
            ends = []
            keeps = []
            sidx = 0
            for key, sg in eg.reference.spans.items():
                if not matches_coref_prefix(self.output_prefix, key):
                    continue
                for ii, mention in enumerate(sg):
                    sidx += 1
                    # convert to span in pred
                    sch, ech = (mention.start_char, mention.end_char)
                    span = eg.predicted.char_span(sch, ech)
                    # TODO add to errors.py
                    if span is None:
                        warnings.warn(
                            "Could not align gold span in span resolver, skipping"
                        )
                        continue
                    starts.append(span.start)
                    ends.append(span.end - 1)
                    keeps.append(sidx - 1)

            starts_xp = self.model.ops.xp.asarray(starts)
            ends_xp = self.model.ops.xp.asarray(ends)
            # span_scores is a Floats3d. Axes: mention x token x start/end
            start_scores = span_scores[:, :, 0][keeps]
            end_scores = span_scores[:, :, 1][keeps]

            n_classes = start_scores.shape[1]
            start_probs = ops.softmax(start_scores, axis=1)
            end_probs = ops.softmax(end_scores, axis=1)
            start_targets = to_categorical(starts_xp, n_classes)
            end_targets = to_categorical(ends_xp, n_classes)
            start_grads = start_probs - start_targets
            end_grads = end_probs - end_targets
            # now return to original shape, with 0s
            final_start_grads = ops.alloc2f(*span_scores[:, :, 0].shape)
            final_start_grads[keeps] = start_grads
            final_end_grads = ops.alloc2f(*final_start_grads.shape)
            final_end_grads[keeps] = end_grads
            # XXX Note this only works with fake batching
            grads = ops.xp.stack((final_start_grads, final_end_grads), axis=2)

            loss = float((grads**2).sum())
        return loss, grads

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Optional[Language] = None,
    ) -> None:
        """Initialize the pipe for training, using a representative set
        of data examples.

        get_examples (Callable[[], Iterable[Example]]): Function that
            returns a representative sample of gold-standard Example objects.
        nlp (Language): The current nlp object the component is part of.

        DOCS: https://spacy.io/api/span_resolver#initialize
        """
        validate_get_examples(get_examples, "SpanResolver.initialize")

        X = []
        Y = []
        for ex in islice(get_examples(), 2):
            X.append(ex.predicted)
            Y.append(ex.reference)

        assert len(X) > 0, Errors.E923.format(name=self.name)
        self.model.initialize(X=X, Y=Y)

        # Store the input dimensionality. nI and nO are not stored explicitly
        # for PyTorch models. This makes it tricky to reconstruct the model
        # during deserialization. So, besides storing the labels, we also
        # store the number of inputs.
        span_resolver = self.model.get_ref("span_resolver")
        self.cfg["nI"] = span_resolver.get_dim("nI")

    def from_bytes(self, bytes_data, *, exclude=tuple()):
        deserializers = {
            "cfg": lambda b: self.cfg.update(srsly.json_loads(b)),
            "vocab": lambda b: self.vocab.from_bytes(b, exclude=exclude),
        }
        from_bytes(bytes_data, deserializers, exclude)

        self._initialize_before_deserializing()

        model_deserializers = {
            "model": lambda b: self.model.from_bytes(b),
        }
        from_bytes(bytes_data, model_deserializers, exclude)

        return self

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
        from_disk(path, deserializers, exclude)

        self._initialize_before_deserializing()

        model_deserializers = {
            "model": load_model,
        }
        from_disk(path, model_deserializers, exclude)

        return self

    def _initialize_before_deserializing(self):
        # The PyTorch model is constructed lazily, so we need to
        # explicitly initialize the model before deserialization.
        model = self.model.get_ref("span_resolver")
        if model.has_dim("nI") is None:
            model.set_dim("nI", self.cfg["nI"])
        self.model.initialize()
