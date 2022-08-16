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
[components.experimental_char_ner_tokenizer]
factory = "experimental_char_ner_tokenizer"
```

## Components

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

```ini
[components.experimental_arc_predicter]
factory = "experimental_arc_predicter"

[components.experimental_arc_labeler]
factory = "experimental_arc_labeler"
```

The arc predicter requires that a previous component (such as `senter`) sets
sentence boundaries during training. Therefore, such a component must be
added to `annotating_components`:

```ini
[training]
annotating_components = ["senter"]
```

The [biaffine parser sample project](projects/biaffine_parser) provides an
example biaffine parser pipeline.

### Span Finder

The SpanFinder is a new experimental component that identifies span boundaries
by tagging potential start and end tokens. It's an ML approach to suggest
candidate spans with higher precision.

`SpanFinder` uses the following parameters:

- `threshold`: Probability threshold for predicted spans.
- `predicted_key`: Name of the [SpanGroup](https://spacy.io/api/spangroup) the predicted spans are saved to.
- `training_key`: Name of the [SpanGroup](https://spacy.io/api/spangroup) the training spans are read from.
- `max_length`: Max length of the predicted spans. No limit when set to `0`. Defaults to `0`.
- `min_length`: Min length of the predicted spans. No limit when set to `0`. Defaults to `0`.

Here is a config excerpt for the `SpanFinder` together with a `SpanCategorizer`:

```ini
[nlp]
lang = "en"
pipeline = ["tok2vec","span_finder","spancat"]
batch_size = 128
disabled = []
before_creation = null
after_creation = null
after_pipeline_creation = null
tokenizer = {"@tokenizers":"spacy.Tokenizer.v1"}

[components]

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.Tok2Vec.v1"

[components.tok2vec.model.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = ${components.tok2vec.model.encode.width}
attrs = ["ORTH", "SHAPE"]
rows = [5000, 2500]
include_static_vectors = false

[components.tok2vec.model.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 96
depth = 4
window_size = 1
maxout_pieces = 3

[components.span_finder]
factory = "experimental_span_finder"
threshold = 0.35
predicted_key = "span_candidates"
training_key = ${vars.spans_key}
min_length = 0
max_length = 0

[components.span_finder.scorer]
@scorers = "spacy-experimental.span_finder_scorer.v1"
predicted_key = ${components.span_finder.predicted_key}
training_key = ${vars.spans_key}

[components.span_finder.model]
@architectures = "spacy-experimental.SpanFinder.v1"

[components.span_finder.model.scorer]
@layers = "spacy.LinearLogistic.v1"
nO=2

[components.span_finder.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = ${components.tok2vec.model.encode.width}

[components.spancat]
factory = "spancat"
max_positive = null
spans_key = ${vars.spans_key}
threshold = 0.5

[components.spancat.model]
@architectures = "spacy.SpanCategorizer.v1"

[components.spancat.model.reducer]
@layers = "spacy.mean_max_reducer.v1"
hidden_size = 128

[components.spancat.model.scorer]
@layers = "spacy.LinearLogistic.v1"
nO = null
nI = null

[components.spancat.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = ${components.tok2vec.model.encode.width}

[components.spancat.suggester]
@misc = "spacy-experimental.span_finder_suggester.v1"
predicted_key = ${components.span_finder.predicted_key}
```

This package includes a [spaCy project](./projects/span_finder) which shows how to train and use the `SpanFinder` together with `SpanCategorizer`.

### Coreference Components

The [CoreferenceResolver](https://spacy.io/api/coref) and [SpanResolver](https://spacy.io/api/span-resolver) are designed to be used together to build a corerefence pipeline, which allows you to identify which spans in a document refer to the same thing. Each component also includes an architecture and scorer. For more details, see their pages in the main spaCy docs.

For an example of how to build a pipeline with the components, see the [example coref project](https://github.com/explosion/projects/tree/v3/experimental/coref).

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

### Misc

Suggester functions for spancat:

**Subtree suggester**: Uses dependency annotation to suggest tokens with their syntactic descendants.

- `spacy-experimental.subtree_suggester.v1`
- `spacy-experimental.ngram_subtree_suggester.v1`

**Chunk suggester**: Suggests noun chunks using the noun chunk iterator, which requires POS and dependency annotation.

- `spacy-experimental.chunk_suggester.v1`
- `spacy-experimental.ngram_chunk_suggester.v1`

**Sentence suggester**: Uses sentence boundaries to suggest sentence spans.

- `spacy-experimental.sentence_suggester.v1`
- `spacy-experimental.ngram_sentence_suggester.v1`

The package also contains a [`merge_suggesters`](spacy_experimental/span_suggesters/merge_suggesters.py) function which can be used to combine suggestions from multiple suggesters.

Here are two config excerpts for using the `subtree suggester` with and without the ngram functionality:

```
[components.spancat.suggester]
@misc = "spacy-experimental.subtree_suggester.v1"
```

```
[components.spancat.suggester]
@misc = "spacy-experimental.ngram_subtree_suggester.v1"
sizes = [1, 2, 3]
```

Note that all the suggester functions are registered in `@misc`.

## Bug reports and issues

Please report bugs in the [spaCy issue
tracker](https://github.com/explosion/spaCy/issues) or open a new thread on the
[discussion board](https://github.com/explosion/spaCy/discussions) for other
issues.

## Older documentation

See the READMEs in earlier [tagged
versions](https://github.com/explosion/spacy-experimental/tags) for details
about components in earlier releases.
