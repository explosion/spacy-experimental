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

span_finder_default_config = """
[model]
@architectures = "spacy-experimental.SpanFinder.v1"

[model.scorer]
@layers = "spacy.LinearLogistic.v1"
nO = 2

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

DEFAULT_SPAN_FINDER_MODEL = Config().from_str(span_finder_default_config)["model"]
DEFAULT_CANDIDATES_KEY = "span_candidates"


@Language.factory(
    "experimental_span_finder",
    assigns=["doc.spans"],
    default_config={
        "threshold": 0.3,
        "model": DEFAULT_SPAN_FINDER_MODEL,
        "candidates_key": DEFAULT_CANDIDATES_KEY,
        "max_length": 0,
        "min_length": 0,
        "scorer": {
            "@scorers": "spacy-experimental.span_finder_scorer.v1",
            "candidates_key": DEFAULT_CANDIDATES_KEY,
        },
    },
    default_score_weights={
        "span_finder_f": 1.0,
        "span_finder_p": 0.0,
        "span_finder_r": 0.0,
    },
)
def make_span_finder(
    nlp: Language,
    name: str,
    model: Model[List[Doc], Floats2d],
    scorer: Optional[Callable],
    threshold: float,
    max_length: int,
    min_length: int,
    candidates_key: str = DEFAULT_CANDIDATES_KEY,
) -> "SpanFinder":
    """Create a SpanFinder component. The component predicts whether a token is the start or the end of a potential span.
    model (Model[List[Doc], Floats2d]): A model instance that
        is given a list of documents and predicts a probability for each token.
    threshold (float): Minimum probability to consider a prediction positive.
    candidates_key (str): Name of the SpanGroup the predicted spans are saved to
    max_length (int): Max length of the produced spans (no max limitation when set to 0)
    min_length (int): Min length of the produced spans (no min limitation when set to 0)
    """
    return SpanFinder(
        nlp.vocab,
        model=model,
        threshold=threshold,
        name=name,
        scorer=scorer,
        max_length=max_length,
        min_length=min_length,
        candidates_key=candidates_key,
    )


def make_span_finder_scorer(candidates_key: str = DEFAULT_CANDIDATES_KEY):
    def span_finder_score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:

        references = get_span_references([doc.reference for doc in examples])
        predictions = get_span_predictions(
            [doc.predicted for doc in examples], candidates_key
        )

        scorer = PRFScore()

        for prediction_doc, reference_doc in zip(predictions, references):
            for prediction, reference in zip(prediction_doc, reference_doc):
                if prediction in reference_doc:
                    scorer.tp += 1
                else:
                    scorer.fp += 1

                if reference not in prediction_doc:
                    scorer.fn += 1

        # Assemble final result
        final_scores: Dict[str, Any] = {
            f"span_finder_f": scorer.fscore,
            f"span_finder_r": scorer.precision,
            f"span_finder_p": scorer.recall,
        }

        return final_scores

    return span_finder_score


def get_span_predictions(docs, candidates_key: str) -> Floats2d:
    """Create a list of predicted spans for scoring"""
    doc_spans = []
    for doc in docs:
        spans = set()
        for span in doc.spans[candidates_key]:
            spans.add((span.start, span.end))
        doc_spans.append(spans)
    return doc_spans


def get_span_references(docs) -> Floats2d:
    """Create a list of reference spans for scoring"""
    doc_spans = []
    for doc in docs:
        spans = set()
        for span_key in doc.spans:
            for span in doc.spans[span_key]:
                spans.add((span.start, span.end))
        doc_spans.append(spans)
    return doc_spans


class SpanFinder(TrainablePipe):
    """Pipeline that learns span boundaries"""

    def __init__(
        self,
        nlp: Language,
        model: Model[List[Doc], Floats2d],
        name: str = "span_finder",
        *,
        threshold: float = 0.5,
        max_length: int = 0,
        min_length: int = 0,
        scorer: Optional[Callable],
        candidates_key: str = "span_finder_candidates",
    ) -> None:
        """Initialize the span boundary detector.
        model (thinc.api.Model): The Thinc Model powering the pipeline component.
        name (str): The component instance name, used to add entries to the
            losses during training.
        threshold (float): Minimum probability to consider a prediction
            positive.
        scorer (Optional[Callable]): The scoring method.
        candidates_key (str): Name of the SpanGroup the span candidates are saved to
        max_length (int): Max length of the produced spans (unlimited when set to 0)
        min_length (int): Min length of the produced spans (unlimited when set to 0)
        """
        self.vocab = nlp
        self.cfg = {
            "threshold": threshold,
            "max_length": max_length,
            "min_length": min_length,
            "candidates_key": candidates_key,
        }
        self.model = model
        self.name = name
        self.scorer = scorer

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
        scores: The scores to set, produced by SpanFinder predict method.
        """
        lengths = [len(doc) for doc in docs]

        offset = 0
        scores_per_doc = []
        for length in lengths:
            scores_per_doc.append(scores[offset : offset + length])
            offset += length

        for doc, doc_scores in zip(docs, scores_per_doc):
            doc.spans[self.cfg["candidates_key"]] = []
            starts = []
            ends = []

            for token, token_score in zip(doc, doc_scores):
                if token_score[0] >= self.cfg["threshold"]:
                    starts.append(token.i)
                if token_score[1] >= self.cfg["threshold"]:
                    ends.append(token.i)

            for start in starts:
                for end in ends:
                    if start < end + 1:
                        span_length = end + 1 - start
                        if (
                            self.cfg["min_length"] <= 0
                            or span_length >= self.cfg["min_length"]
                        ) and (
                            self.cfg["max_length"] <= 0
                            or span_length <= self.cfg["max_length"]
                        ):
                            doc.spans[self.cfg["candidates_key"]].append(
                                doc[start : end + 1]
                            )

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
        sgd (Optional[thinc.api.Optimizer]): The optimizer.
        losses (Optional[Dict[str, float]]): Optional record of the loss during training.
            Updated using the component name as the key.
        RETURNS (Dict[str, float]): The updated losses dictionary.
        """
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        predicted = [eg.predicted for eg in examples]
        set_dropout_rate(self.model, drop)
        scores, backprop_scores = self.model.begin_update(predicted)
        loss, d_scores = self.get_loss(examples, scores)
        backprop_scores(d_scores)
        if sgd is not None:
            self.finish_update(sgd)
        losses[self.name] += loss
        return losses

    def get_loss(self, examples, scores) -> Tuple[float, float]:
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores.
        examples (Iterable[Examples]): The batch of examples.
        scores: Scores representing the model's predictions.
        RETURNS (Tuple[float, float]): The loss and the gradient.
        """
        references = [eg.reference for eg in examples]
        reference_results = self.model.ops.asarray(
            self._get_reference(references), dtype=float32
        )
        d_scores = scores - reference_results
        loss = float((d_scores**2).sum())
        return loss, d_scores

    def _get_reference(self, docs) -> Floats2d:
        """Create a reference list of token probabilities for calculating loss"""
        reference_results = []
        for doc in docs:
            start_indices = []
            end_indices = []

            for spankey in doc.spans:
                for span in doc.spans[spankey]:
                    start_indices.append(span.start)
                    end_indices.append(span.end - 1)

            for token in doc:
                is_start = 0
                is_end = 0
                if token.i in start_indices:
                    is_start = 1
                if token.i in end_indices:
                    is_end = 1
                reference_results.append((is_start, is_end))

        return reference_results

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
        """
        subbatch: List[Example] = []

        for eg in get_examples():
            subbatch.append(eg)

        if subbatch:
            docs = [eg.reference for eg in subbatch]
            Y = self.model.ops.asarray(self._get_reference(docs), dtype=float32)
            self.model.initialize(X=docs, Y=Y)
        else:
            self.model.initialize()
