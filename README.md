<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spacy-biaffine-parser

An experimental biaffine parser for spaCy v3.2+.

## Install

Install spaCy 3.2.0 or later:

```bash
pip install spacy
```

Install from source:

```bash
pip install -U pip setuptools wheel
pip install .
```

Or from the repo URL:

```bash
pip install -U pip setuptools wheel
pip install https://github.com/danieldk/spacy-biaffine-parser/archive/main.zip
```

## Usage

Once this package is installed, the biaffine parser is registered as a spaCy
component factory, so you can specify it like this in your config:

```ini
[components.arc_predicter]
factory = "arc_predicter"

[components.arc_labeler]
factory = "arc_labeler"
```

Or start from a blank model in python:

```python
import spacy

nlp = spacy.blank("en")
nlp.add_pipe("arc_predicter")
nlp.add_pipe("arc_labeler")
```

## Demo Project

See training examples in the [`demo project`](project).
