<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spacy-experimental: Cutting-edge experimental spaCy components and features

This package includes experimental components and features for
[spaCy](https://spacy.io) v3.x, for example model architectures, pipeline
components and utilities.

[![Azure Pipelines](https://img.shields.io/azure-devops/build/explosion-ai/public/26/master.svg?logo=azure-pipelines&style=flat-square&label=build)](https://dev.azure.com/explosion-ai/public/_build?definitionId=26)
[![pypi Version](https://img.shields.io/pypi/v/spacy-experimental.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/spacy-experimental/)

## Installation

Install with `pip`:

```bash
python -m pip install -U pip setuptools wheel
python -m pip install spacy-experimental
```

## Using spacy-experimental

Components and features may be modified or removed in any release, so always
specify the exact version as a package requirement if you're experimenting with
a particular component, e.g.:

```
spacy-experimental==0.147.0
```

Then you can add the experimental components to your config or import from
`spacy_experimental`:

```ini
[components.experimental_edit_tree_lemmatizer]
factory = "experimental_edit_tree_lemmatizer"
```

## Components

### Edit tree lemmatizer

```ini
[components.experimental_edit_tree_lemmatizer]
factory = "experimental_edit_tree_lemmatizer"
# token attr to use as backoff with the predicted trees are not applicable; null to leave unset
backoff = "orth"
# prune trees that are applied less than this frequency in the training data
min_tree_freq = 2
# whether to overwrite existing lemma annotation
overwrite = false
scorer = {"@scorers":"spacy.lemmatizer_scorer.v1"}
# try to apply at most the k most probable edit trees
top_k = 1
```

### Trainable character-based tokenizers

Two trainable tokenizers represent tokenization as a sequence tagging problem
over individual characters and use the existing spaCy tagger and NER
architectures to perform the tagging.

In the spaCy pipeline, a simple "pretokenizer" is applied as the pipeline
tokenizer to split each doc into individual characters and the trainable
tokenizer is a pipeline component that retokenizes the doc. The pretokenizer
needs to be configured manually in the config or with `spacy.blank()`:

```python
nlp = spacy.blank(
    "en",
    config={
        "nlp": {
            "tokenizer": {"@tokenizers": "spacy-experimental.char_pretokenizer.v1"}
        }
    },
)
```

The two tokenizers currently reset any existing tag or entity annotation
respectively in the process of retokenizing.

#### Character-based tagger tokenizer

In the tagger version `experimental_char_tagger_tokenizer`, the tagging problem
is represented internally with character-level tags for token start (`T`),
token internal (`I`), and outside a token (`O`). This representation comes from
[Elephant: Sequence Labeling for Word and Sentence
Segmentation](https://aclanthology.org/D13-1146/) (Evang et al., 2013).

```none
This is a sentence.
TIIIOTIOTOTIIIIIIIT
```

With the option `annotate_sents`, `S` replaces `T` for the first token in each
sentence and the component predicts both token and sentence boundaries.

```none
This is a sentence.
SIIIOTIOTOTIIIIIIIT
```

A config excerpt for `experimental_char_tagger_tokenizer`:

```ini
[nlp]
pipeline = ["experimental_char_tagger_tokenizer"]
tokenizer = {"@tokenizers":"spacy-experimental.char_pretokenizer.v1"}

[components]

[components.experimental_char_tagger_tokenizer]
factory = "experimental_char_tagger_tokenizer"
annotate_sents = true
scorer = {"@scorers":"spacy-experimental.tokenizer_senter_scorer.v1"}

[components.experimental_char_tagger_tokenizer.model]
@architectures = "spacy.Tagger.v1"
nO = null

[components.experimental_char_tagger_tokenizer.model.tok2vec]
@architectures = "spacy.Tok2Vec.v2"

[components.experimental_char_tagger_tokenizer.model.tok2vec.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = 128
attrs = ["ORTH","LOWER","IS_DIGIT","IS_ALPHA","IS_SPACE","IS_PUNCT"]
rows = [1000,500,50,50,50,50]
include_static_vectors = false

[components.experimental_char_tagger_tokenizer.model.tok2vec.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 128
depth = 4
window_size = 4
maxout_pieces = 2
```

#### Character-based NER tokenizer

In the NER version, each character in a token is part of an entity:

```none
T	B-TOKEN
h	I-TOKEN
i	I-TOKEN
s	I-TOKEN
 	O
i	B-TOKEN
s	I-TOKEN
	O
a	B-TOKEN
 	O
s	B-TOKEN
e	I-TOKEN
n	I-TOKEN
t	I-TOKEN
e	I-TOKEN
n	I-TOKEN
c	I-TOKEN
e	I-TOKEN
.	B-TOKEN
```

A config excerpt for `experimental_char_ner_tokenizer`:

```ini
[nlp]
pipeline = ["experimental_char_ner_tokenizer"]
tokenizer = {"@tokenizers":"spacy-experimental.char_pretokenizer.v1"}

[components]

[components.experimental_char_ner_tokenizer]
factory = "experimental_char_ner_tokenizer"
scorer = {"@scorers":"spacy-experimental.tokenizer_scorer.v1"}

[components.experimental_char_ner_tokenizer.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = true
nO = null

[components.experimental_char_ner_tokenizer.model.tok2vec]
@architectures = "spacy.Tok2Vec.v2"

[components.experimental_char_ner_tokenizer.model.tok2vec.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = 128
attrs = ["ORTH","LOWER","IS_DIGIT","IS_ALPHA","IS_SPACE","IS_PUNCT"]
rows = [1000,500,50,50,50,50]
include_static_vectors = false

[components.experimental_char_ner_tokenizer.model.tok2vec.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 128
depth = 4
window_size = 4
maxout_pieces = 2
```

The NER version does not currently support sentence boundaries, but it would be
easy to extend using a `B-SENT` entity type.

### Biaffine parser

A biaffine dependency parser, similar to that proposed in [Deep Biaffine
Attention for Neural Dependency Parsing](Deep Biaffine Attention for Neural
Dependency Parsing) (Dozat & Manning, 2016). The parser consists of two parts:
an edge predicter and an edge labeler. For example:

``` ini
[components.experimental_arc_predicter]
factory = "experimental_arc_predicter"

[components.experimental_arc_labeler]
factory = "experimental_arc_labeler"
```

The arc predicter requires that a previous component (such as `senter`) sets
sentence boundaries during training. Therefore, such a component must be
added to `annotating_components`:

``` ini
[training]
annotating_components = ["senter"]
```

The [biaffine parser sample project](projects/biaffine_parser) provides an
example biaffine parser pipeline.

## Architectures

None currently.

## Other

### Tokenizers

- `spacy-experimental.char_pretokenizer.v1`: Tokenize a text into individual
  characters.

### Scorers

- `spacy-experimental.tokenizer_scorer.v1`: Score tokenization.
- `spacy-experimental.tokenizer_senter_scorer.v1`: Score tokenization and
  sentence segmentation.

## Bug reports and issues

Please report bugs in the [spaCy issue
tracker](https://github.com/explosion/spaCy/issues) or open a new thread on the
[discussion board](https://github.com/explosion/spaCy/discussions) for other
issues.

## Older documentation

See the READMEs in earlier [tagged
versions](https://github.com/explosion/spacy-experimental/tags) for details
about components in earlier releases.
