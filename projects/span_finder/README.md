<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Span Finder

The SpanFinder is a new experimental component that identifies span boundaries
by tagging potential start and end tokens. It's an ML approach to suggest
candidate spans with higher precision.

This project shows how to use the `SpanFinder` together with a `SpanCategorizer`.

## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `install` | Install requirements |
| `preprocess_healthsea` | Format Healthsea annotations into .spaCy training format |
| `preprocess_genia` | Format Genia annotations into .spaCy training format |
| `preprocess_toxic` | Format annotations into .spaCy training format |
| `train_spancat` | Train a spancat model on the `dataset` defined in `project.yml` |
| `evaluate_spancat` | Evaluate a trained spancat model  on the `dataset` defined in `project.yml` |
| `reset` | Reset the project to its original state and delete all training process |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `preprocess` | `preprocess_healthsea` &rarr; `preprocess_toxic` &rarr; `preprocess_genia` |
| `spancat` | `train_spancat` &rarr; `evaluate_spancat` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/healthsea_ner.jsonl` | URL | Annotations from the Healthsea dataset |
| `assets/toxic_spans_annotations.csv` | URL | Annotations from the ToxicSpans dataset |
| `assets/toxic_spans.csv` | URL | Spans from the ToxicSpans dataset |
| `assets/toxic_spans_comments.csv` | URL | Comments from the ToxicSpans dataset |
| `assets/genia_train.iob` | URL | Training annotations from the Genia dataset |
| `assets/genia_dev.iob` | URL | Dev annotations from the Genia dataset |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

### 🔧 Parameters

| Parameter    | Description                                                                                |
| ------------ | ------------------------------------------------------------------------------------------ |
| `config`     | Choose between a config with Tok2Vec embedding and Transformer (roberta-base) embedding    |
| `dataset`    | Choose between three datasets (Healthsea, ToxicSpans, and Genia)                           |
| `suggester`  | Choose between two suggester architectures (SpanFinder, Ngram)                             |
| `train`      | Choose a filename for your training data                                                   |
| `dev`        | Choose a filename for your development data                                                |
| `spans_key`   | Choose a key to specify the SpanGroup for the spancat component to save the predictions to |
| `gpu_id`     | Choose whether you want to use your GPU (device number) or CPU (-1)                        |
| `eval_split` | Choose an evaluation split for the dataset (Only affects the Healthsea dataset)            |
