from typing import Iterable, Optional, Dict, Callable, Any, List
from functools import partial
import warnings

from thinc.types import Floats2d, Ints2d
from thinc.api import Model, Config, Optimizer
from thinc.api import set_dropout_rate
from itertools import islice
import srsly

from spacy.pipeline import TrainablePipe
from spacy.language import Language
from spacy.training import Example, validate_examples, validate_get_examples
from spacy.errors import Errors
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.util import from_disk, from_bytes

from .coref_util import create_gold_scores, MentionClusters, create_head_span_idxs
from .coref_util import get_clusters_from_doc, get_predicted_clusters
from .coref_util import DEFAULT_CLUSTER_PREFIX

from .coref_scorer import score_coref_clusters


default_config = """
[model]
@architectures = "spacy-experimental.Coref.v1"
distance_embedding_size = 20
hidden_size = 1024
depth = 1
dropout = 0.3
antecedent_limit = 50
antecedent_batch_size = 512

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
DEFAULT_COREF_MODEL = Config().from_str(default_config)["model"]


def coref_scorer(
    examples: Iterable[Example], span_cluster_prefix: str, **kwargs
) -> Dict[str, Any]:
    return score_coref_clusters(
        examples, span_cluster_prefix=span_cluster_prefix, **kwargs
    )


def make_coref_scorer(span_cluster_prefix: str = DEFAULT_CLUSTER_PREFIX):
    return partial(coref_scorer, span_cluster_prefix=span_cluster_prefix)


@Language.factory(
    "experimental_coref",
    assigns=["doc.spans"],
    requires=["doc.spans"],
    default_config={
        "model": DEFAULT_COREF_MODEL,
        "span_cluster_prefix": DEFAULT_CLUSTER_PREFIX,
        "scorer": {
            "@scorers": "spacy-experimental.coref_scorer.v1",
            "span_cluster_prefix": DEFAULT_CLUSTER_PREFIX,
        },
    },
    default_score_weights={"coref_f": 1.0, "coref_p": None, "coref_r": None},
)
def make_coref(
    nlp: Language,
    name: str,
    model: Model[List[Doc], Floats2d],
    scorer: Optional[Callable],
    span_cluster_prefix: str,
) -> "CoreferenceResolver":
    """Create a CoreferenceResolver component."""

    return CoreferenceResolver(
        nlp.vocab, model, name, span_cluster_prefix=span_cluster_prefix, scorer=scorer
    )


class CoreferenceResolver(TrainablePipe):
    """Pipeline component for coreference resolution.

    DOCS: https://spacy.io/api/coref
    """

    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "coref",
        *,
        span_mentions: str = "coref_mentions",
        span_cluster_prefix: str = DEFAULT_CLUSTER_PREFIX,
        scorer: Optional[Callable] = coref_scorer,
    ) -> None:
        """Initialize a coreference resolution component.

        vocab (Vocab): The shared vocabulary.
        model (thinc.api.Model): The Thinc Model powering the pipeline component.
        name (str): The component instance name, used to add entries to the
            losses during training.
        span_mentions (str): Key in doc.spans where the candidate coref mentions
            are stored in.
        span_cluster_prefix (str): Prefix for the key in doc.spans to store the
            coref clusters in.
        scorer (Optional[Callable]): The scoring method. Defaults to
            Scorer.score_coref_clusters.

        DOCS: https://spacy.io/api/coref#init
        """
        self.vocab = vocab
        self.model = model
        self.name = name
        self.span_mentions = span_mentions
        self.span_cluster_prefix = span_cluster_prefix
        self._rehearsal_model = None

        self.cfg: Dict[str, Any] = {}
        self.scorer = scorer

    def predict(self, docs: Iterable[Doc]) -> List[MentionClusters]:
        """Apply the pipeline's model to a batch of docs, without modifying them.
        Return the list of predicted clusters.

        docs (Iterable[Doc]): The documents to predict.
        RETURNS (List[MentionClusters]): The model's prediction for each document.

        DOCS: https://spacy.io/api/coref#predict
        """
        out = []
        for doc in docs:
            scores, idxs = self.model.predict([doc])
            # idxs is a list of mentions (start / end idxs)
            # each item in scores includes scores and a mapping from scores to mentions
            ant_idxs = idxs

            # TODO batching
            xp = self.model.ops.xp

            starts = xp.arange(0, len(doc))
            ends = xp.arange(0, len(doc)) + 1

            predicted = get_predicted_clusters(xp, starts, ends, ant_idxs, scores)
            out.append(predicted)

        return out

    def set_annotations(
        self, docs: Iterable[Doc], clusters_by_doc: List[MentionClusters]
    ) -> None:
        """Modify a batch of Doc objects, using pre-computed scores.

        docs (Iterable[Doc]): The documents to modify.
        clusters: The span clusters, produced by CoreferenceResolver.predict.

        DOCS: https://spacy.io/api/coref#set_annotations
        """
        docs = list(docs)
        if len(docs) != len(clusters_by_doc):
            raise ValueError(
                "Found coref clusters incompatible with the "
                "documents provided to the 'coref' component. "
                "This is likely a bug in spaCy."
            )

        for doc, clusters in zip(docs, clusters_by_doc):
            for ii, cluster in enumerate(clusters, 1):
                key = f"{self.span_cluster_prefix}_{ii}"
                if key in doc.spans:
                    raise ValueError(
                        f'Existing spans with prefix "{key}" found, not '
                        "overwriting. If you run more than one instance of coref "
                        "you should use different target span prefixes."
                    )

                doc.spans[key] = []
                for mention in cluster:
                    doc.spans[key].append(doc[mention[0] : mention[1]])

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

        DOCS: https://spacy.io/api/coref#update
        """
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        validate_examples(examples, "CoreferenceResolver.update")
        if not any(len(eg.predicted) if eg.predicted else 0 for eg in examples):
            # Handle cases where there are no tokens in any docs.
            return losses
        set_dropout_rate(self.model, drop)

        total_loss = 0

        for eg in examples:
            if eg.x.text != eg.y.text:
                # TODO assign error number
                raise ValueError(
                    """Text, including whitespace, must match between reference and
                    predicted docs in coref training.
                    """
                )
            preds, backprop = self.model.begin_update([eg.predicted])
            score_matrix, mention_idx = preds
            loss, d_scores = self.get_loss([eg], score_matrix, mention_idx)
            total_loss += loss
            backprop((d_scores, mention_idx))

        if sgd is not None:
            self.finish_update(sgd)
        losses[self.name] += total_loss
        return losses

    def rehearse(self, examples, *, sgd=None, losses=None, **config):
        # TODO this should be added later
        raise NotImplementedError(
            Errors.E931.format(
                parent="CoreferenceResolver", method="add_label", name=self.name
            )
        )

    def add_label(self, label: str) -> int:
        """Technically this method should be implemented from TrainablePipe,
        but it is not relevant for the coref component.
        """
        raise NotImplementedError(
            Errors.E931.format(
                parent="CoreferenceResolver", method="add_label", name=self.name
            )
        )

    def get_loss(
        self,
        examples: Iterable[Example],
        score_matrix: Floats2d,
        mention_idx: Ints2d,
    ):
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores.

        examples (Iterable[Examples]): The batch of examples.
        score_matrix (Floats2d): Scores representing the model's predictions.
        mention_idx (Ints2d): Mention indices.
        RETURNS (Tuple[float, float]): The loss and the gradient.

        DOCS: https://spacy.io/api/coref#get_loss
        """
        ops = self.model.ops
        xp = ops.xp

        examples = list(examples)
        if len(examples) > 1:
            # TODO handle more than one document
            raise ValueError(
                "Got more than one example, but only fake batching is supported."
            )
        example = examples[0]

        clusters_by_char = get_clusters_from_doc(
            example.reference, use_heads=True, prefix=self.span_cluster_prefix
        )
        # convert to token clusters, and give up if necessary
        clusters = []
        for cluster in clusters_by_char:
            cc = []
            for start_char, end_char in cluster:
                span = example.predicted.char_span(start_char, end_char)
                if span is None:
                    # TODO log more details
                    raise IndexError(
                        "Misalignment in Coref. Head token has no match in training doc."
                    )
                cc.append((span.start, span.end))
            clusters.append(cc)

        span_idxs = create_head_span_idxs(ops, len(example.predicted))
        gscores = create_gold_scores(span_idxs, clusters)
        # Ensure that the returned array has the same backend as the model.
        gscores = ops.asarray2f(gscores)  # type: ignore
        top_gscores = xp.take_along_axis(gscores, mention_idx, axis=1)
        # now add the placeholder
        gold_placeholder = ~top_gscores.any(axis=1).T
        gold_placeholder = xp.expand_dims(gold_placeholder, 1)
        top_gscores = xp.concatenate((gold_placeholder, top_gscores), 1)

        with warnings.catch_warnings():
            # This builds a mask by building a matrix of 0/1 values and taking
            # the log. log(0) gives a warning in numpy, but we're doing it on
            # purpose to get infinite values, so it's safe to ignore.
            # Note cupy gives no warning.
            warnings.filterwarnings(
                "ignore", message="divide by zero", category=RuntimeWarning
            )
            log_marg = ops.softmax(score_matrix + ops.xp.log(top_gscores), axis=1)
        log_norm = ops.softmax(score_matrix, axis=1)
        grad = log_norm - log_marg
        loss = float((grad**2).sum())

        return loss, grad

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

        DOCS: https://spacy.io/api/coref#initialize
        """
        validate_get_examples(get_examples, "CoreferenceResolver.initialize")

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
        coref_clusterer = self.model.get_ref("coref_clusterer")
        self.cfg["nI"] = coref_clusterer.get_dim("nI")

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
        model = self.model.get_ref("coref_clusterer")
        if model.has_dim("nI") is None:
            model.set_dim("nI", self.cfg["nI"])
        self.model.initialize()
