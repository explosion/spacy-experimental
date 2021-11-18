<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spacy-experimental: Cutting-edge experimental spaCy components and features

This package includes experimental components and features for
[spaCy](https://spacy.io) v3.x, for example model architectures, pipeline
components and utilities.

[![Azure Pipelines](https://img.shields.io/azure-devops/build/explosion-ai/public/21/master.svg?logo=azure-pipelines&style=flat-square&label=build)](https://dev.azure.com/explosion-ai/public/_build?definitionId=21)
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
[components.edit_tree_lemmatizer]
factory = "edit_tree_lemmatizer"
```

## Components

### `edit_tree_lemmatizer`

```ini
[components.edit_tree_lemmatizer]
factory = "edit_tree_lemmatizer"
# token attr to use backoff with the predicted trees are not applicable; null to leave unset
backoff = "orth"
# prune trees that are applied less than this frequency in the training data
min_tree_freq = 2
# whether to overwrite existing lemma annotation
overwrite = false
scorer = {"@scorers":"spacy.lemmatizer_scorer.v1"}
# try to apply at most the k most probably edit trees
top_k = 1
```

## Architectures

None currently.

## Other

None currently.

## Older documentation

See the READMEs in earlier [tagged versions](/tags) for details about
components in earlier releases.
