<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Span Boundary Detection

This project introduces a new experimental suggester that learns to predict span boundaries for more precise candidate spans.

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
| `preprocess_healthsea` | Format Healthsea annotations into .spaCy training format |
| `preprocess_genia` | Format Genia annotations into .spaCy training format |
| `preprocess_toxic` | Format annotations into .spaCy training format |
| `analyze_healthsea` | Analyze Healthsea training dataset |
| `analyze_toxic` | Analyze ToxicSpans training dataset |
| `analyze_genia` | Analyze Genia training dataset |
| `train_sbd` | Train SpanBoundaryDetection model |
| `evaluate_sbd` | Evaluate a trained SpanBoundaryDetection model |
| `train_spancat` | Train a spancat model |
| `evaluate_spancat` | Evaluate a trained spancat model |
| `evaluate_suggester` | Evaluate the suggester of a trained spancat model |
| `reset` | Reset the project to its original state and delete all training process |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `preprocess` | `preprocess_healthsea` &rarr; `preprocess_toxic` &rarr; `preprocess_genia` |
| `analyze` | `analyze_healthsea` &rarr; `analyze_toxic` &rarr; `analyze_genia` |
| `train` | `train_sbd` &rarr; `evaluate_sbd` &rarr; `train_spancat` &rarr; `evaluate_spancat` &rarr; `evaluate_suggester` |

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