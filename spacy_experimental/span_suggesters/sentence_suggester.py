from typing import List, Optional, Iterable, cast
from thinc.api import get_current_ops, Ops
from thinc.types import Ragged, Ints1d
from spacy.tokens import Doc
from spacy.util import registry
from spacy.pipeline.spancat import Suggester

@registry.misc("experimental.sentence_suggester.v1")
def build_sentence_suggester(sizes: List[int]) -> Suggester:
    """Suggest ngrams and sentences. Requires sentence boundaries"""

    def sentence_suggester(docs: Iterable[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        if ops is None:
            ops = get_current_ops()
        spans = []
        lengths = []

        if sizes:
            suggester = registry.misc.get("spacy.ngram_suggester.v1")(sizes)

        for doc in docs:
            cache = set()
            length = 0

            # ngram-suggestion
            if sizes:
                ngram_spans = suggester([doc], ops=ops)
                for ngram_span in ngram_spans.data:
                    ngram_span = ops.to_numpy(ngram_span)
                    spans.append((ngram_span[0], ngram_span[1]))
                    cache.add((ngram_span[0], ngram_span[1]))
                    length += 1

            # sentence-suggestion
            for sentence in doc.sents:
                if (sentence.start, sentence.end) not in cache:
                    spans.append((sentence.start, sentence.end))
                    cache.add((sentence.start, sentence.end))
                    length += 1

            lengths.append(length)

        lengths_array = cast(Ints1d, ops.asarray(lengths, dtype="i"))
        if len(spans) > 0:
            output = Ragged(ops.asarray(spans, dtype="i"), lengths_array)
        else:
            output = Ragged(ops.xp.zeros((0, 0), dtype="i"), lengths_array)

        return output

    return sentence_suggester
