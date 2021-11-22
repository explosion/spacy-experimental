from typing import Callable
from spacy.language import Language
from spacy.tokens import Doc
from spacy.util import DummyTokenizer
from spacy.vocab import Vocab
from spacy import util


def char_pretokenizer_v1() -> Callable[[Language], "CharPretokenizer"]:
    """Function to create a character tokenizer.
    """
    def char_pretokenizer_factory(nlp: Language) -> "CharPretokenizer":
        return CharPretokenizer(nlp.vocab)

    return char_pretokenizer_factory


class CharPretokenizer(DummyTokenizer):
    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def __call__(self, string: str):
        # split into individual characters
        words = list(string)
        return Doc(self.vocab, words=words, spaces=[False] * len(words))
