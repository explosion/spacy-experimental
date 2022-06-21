from spacy.tokens import DocBin, Span
import spacy
from wasabi import Printer
import json
from pathlib import Path
import typer

msg = Printer()


def main(
    json_loc: Path,
    train_file: Path,
    dev_file: Path,
    eval_split: float,
    span_key: str,
):
    """Parse the annotations into a training and development set for Spancat."""

    empty_docs = []
    docs = []
    nlp = spacy.blank("en")
    total_span_count = {}
    max_span_length = 0

    msg.info("Processing Healthsea")
    # Load dataset
    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)

            if example["answer"] == "accept":
                doc = nlp(example["text"])
                spans = []

                if "spans" in example:
                    for span in example["spans"]:
                        spans.append(
                            Span(
                                doc,
                                span["token_start"],
                                span["token_end"] + 1,
                                span["label"],
                            )
                        )

                        if span["label"] not in total_span_count:
                            total_span_count[span["label"]] = 0

                        total_span_count[span["label"]] += 1

                        span_length = (span["token_end"] + 1) - span["token_start"]
                        if span_length > max_span_length:
                            max_span_length = span_length

                doc.set_ents(spans)
                doc.spans[span_key] = spans

                if len(doc.ents) > 0:
                    docs.append(doc)
                else:
                    empty_docs.append(doc)

    # Split
    train = []
    dev = []

    split = int(len(docs) * eval_split)
    empty_split = int(len(empty_docs) * eval_split)
    train = docs[split:] + empty_docs[empty_split:]
    dev = docs[:split] + empty_docs[:empty_split]

    # Save to disk
    docbin = DocBin(docs=train, store_user_data=True)
    docbin.to_disk(train_file)

    docbin = DocBin(docs=dev, store_user_data=True)
    docbin.to_disk(dev_file)
    msg.good(f"Processing Healthsea done")


if __name__ == "__main__":
    typer.run(main)
