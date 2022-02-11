from typing import List, Dict, Callable, Tuple, Optional, Iterable, Any
from thinc.api import Config, Model, set_dropout_rate
from thinc.api import Optimizer
from thinc.types import Floats2d
from numpy import float32

from spacy.language import Language
from spacy.util import registry
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.tokens import Doc, Token
from spacy.training import Example
from spacy.scorer import PRFScore

sbd_default_config = """
[model]
@architectures = "spacy.PyTorchSpanBoundaryDetection.v1"
hidden_size = 128

[model.scorer]
@layers = "spacy.LinearLogistic.v1"
nO=2

[model.tok2vec]
@architectures = "spacy.Tok2Vec.v1"

[model.tok2vec.embed]
@architectures = "spacy.MultiHashEmbed.v1"
width = 96
rows = [5000, 2000, 1000, 1000]
attrs = ["ORTH", "PREFIX", "SUFFIX", "SHAPE"]
include_static_vectors = false

[model.tok2vec.encode]
@architectures = "spacy.MaxoutWindowEncoder.v1"
width = ${model.tok2vec.embed.width}
window_size = 1
maxout_pieces = 3
depth = 4
"""

DEFAULT_SBD_MODEL = Config().from_str(sbd_default_config)["model"]


@Language.factory(
    "spacy.SpanBoundaryDetection.v1",
    # Placeholder -> throws error if empty
    assigns=["doc.spans"],
    default_config={
        "threshold": 0.5,
        "model": DEFAULT_SBD_MODEL,
        "scorer": {"@scorers": "spacy.sbd_scorer.v1"},
    },
    default_score_weights={
        "sbd_start_f": 1.0,
        "sbd_start_p": 0.0,
        "sbd_start_r": 0.0,
        "sbd_end_f": 1.0,
        "sbd_end_p": 0.0,
        "sbd_end_r": 0.0,
    },
)
def make_sbd(
    nlp: Language,
    name: str,
    model: Model[List[Doc], Floats2d],
    scorer: Optional[Callable],
    threshold: float,
) -> "SpanBoundaryDetection":
    """Create a SpanBoundaryDetection component. The component predicts whether a token is the start or the end of a span.
    model (Model[List[Doc], Floats2d]): A model instance that
        is given a list of documents and predicts a probability for each token.
    threshold (float): Minimum probability to consider a prediction positive.
    """
    return SpanBoundaryDetection(
        nlp.vocab,
        model=model,
        threshold=threshold,
        name=name,
        scorer=scorer,
    )


@registry.scorers("spacy.sbd_scorer.v1")
def make_sbd_scorer():
    return sbd_score


def sbd_score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:

    references = [doc.reference for doc in examples]
    predictions = [doc.predicted for doc in examples]

    reference_results = get_reference(references)
    prediction_results = get_predictions(predictions)

    scorer_start = PRFScore()
    scorer_end = PRFScore()

    for prediction, reference in zip(prediction_results, reference_results):

        start_prediction = prediction[0]
        end_prediction = prediction[1]
        start_reference = reference[0]
        end_reference = reference[1]

        # Start
        if start_prediction == 1 and start_reference == 1:
            scorer_start.tp += 1
        elif start_prediction == 1 and start_reference == 0:
            scorer_start.fp += 1
        elif start_prediction == 0 and start_reference == 1:
            scorer_start.fn += 1

        # End
        if end_prediction == 1 and end_reference == 1:
            scorer_end.tp += 1
        elif end_prediction == 1 and end_reference == 0:
            scorer_end.fp += 1
        elif end_prediction == 0 and end_reference == 1:
            scorer_end.fn += 1

    # Assemble final result
    final_scores: Dict[str, Any] = {
        f"sbd_start_f": scorer_start.fscore,
        f"sbd_start_p": scorer_start.precision,
        f"sbd_start_r": scorer_start.recall,
        f"sbd_end_f": scorer_end.fscore,
        f"sbd_end_p": scorer_end.precision,
        f"sbd_end_r": scorer_end.recall,
    }

    return final_scores


def get_reference(docs) -> Floats2d:
    """Create a reference list of token probabilities for calculating loss and metrics"""
    reference_results = []
    for doc in docs:
        start_indices = []
        end_indices = []

        for spankey in doc.spans:
            for span in doc.spans[spankey]:
                start_indices.append(span.start)
                end_indices.append(span.end)

        for token in doc:
            is_start = 0
            is_end = 0
            if token.i in start_indices:
                is_start = 1
            if token.i in end_indices:
                is_end = 1
            reference_results.append((is_start, is_end))

    return reference_results


def get_predictions(docs) -> Floats2d:
    """Create a prediction list of token start/end probabilities for evaluation"""
    prediction_results = []
    for doc in docs:
        for token in doc:
            prediction_results.append([token._.span_start, token._.span_end])
    return prediction_results


class SpanBoundaryDetection(TrainablePipe):
    """Pipeline that learns start and end tokens of spans"""

    def __init__(
        self,
        nlp: Language,
        model: Model[List[Doc], Floats2d],
        name: str = "sbd",
        *,
        threshold: float = 0.5,
        scorer: Optional[Callable] = sbd_score,
    ) -> None:
        """Initialize the span boundary detector.
        model (thinc.api.Model): The Thinc Model powering the pipeline component.
        name (str): The component instance name, used to add entries to the
            losses during training.
        threshold (float): Minimum probability to consider a prediction
            positive.
        scorer (Optional[Callable]): The scoring method.
        """
        self.vocab = nlp
        self.cfg = {
            "threshold": threshold,
        }
        self.model = model
        self.name = name
        self.scorer = scorer
        Token.set_extension("span_start", default=0, force=True)
        Token.set_extension("span_end", default=0, force=True)

    def predict(self, docs: Iterable[Doc]):
        """Apply the pipeline's model to a batch of docs, without modifying them.
        docs (Iterable[Doc]): The documents to predict.
        RETURNS: The models prediction for each document.
        """
        scores = self.model.predict(docs)
        return scores

    def set_annotations(self, docs: Iterable[Doc], scores: Floats2d) -> None:
        """Modify a batch of Doc objects, using pre-computed scores.
        docs (Iterable[Doc]): The documents to modify.
        scores: The scores to set, produced by SpanCategorizer.predict.
        """
        lengths = [len(doc) for doc in docs]

        offset = 0
        scores_per_doc = []
        for length in lengths:
            scores_per_doc.append(scores[offset : offset + length])
            offset += length

        for doc, score_doc in zip(docs, scores_per_doc):
            for token, score_token in zip(doc, score_doc):

                if score_token[0] > self.cfg["threshold"]:
                    score_token[0] = 1
                else:
                    score_token[0] = 0

                if score_token[1] > self.cfg["threshold"]:
                    score_token[1] = 1
                else:
                    score_token[1] = 0

                token._.span_start = score_token[0]
                token._.span_end = score_token[1]

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
        """
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        references = [eg.reference for eg in examples]
        set_dropout_rate(self.model, drop)
        scores, backprop_scores = self.model.begin_update(references)
        loss, d_scores = self.get_loss(references, scores)
        backprop_scores(d_scores)
        if sgd is not None:
            self.finish_update(sgd)
        losses[self.name] += loss
        return losses

    def get_loss(self, docs, scores) -> Tuple[float, float]:
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores.
        examples (Iterable[Examples]): The batch of examples.
        scores: Scores representing the model's predictions.
        RETURNS (Tuple[float, float]): The loss and the gradient.
        """
        reference_results = self.model.ops.asarray(get_reference(docs), dtype=float32)
        d_scores = scores - reference_results
        loss = float((d_scores ** 2).sum())
        return loss, d_scores

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
        nlp (Optional[Language]): The current nlp object the component is part of.
        labels (Optional[List[str]]): The labels to add to the component, typically generated by the
            `init labels` command. If no labels are provided, the get_examples
            callback is used to extract the labels from the data.
        """
        subbatch: List[Example] = []

        for eg in get_examples():
            subbatch.append(eg)

        if subbatch:
            docs = [eg.reference for eg in subbatch]
            Y = self.model.ops.asarray(get_reference(docs), dtype=float32)
            self.model.initialize(X=docs, Y=Y)
        else:
            self.model.initialize()
