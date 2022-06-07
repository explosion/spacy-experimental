from spacy.tokens import DocBin, Span
import spacy
import pandas as pd
from spacy.tokens.doc import Doc
from tqdm import tqdm

from wasabi import Printer
from wasabi import table

from pathlib import Path
import typer

msg = Printer()


def main(
    spans_file: Path,
    annotations_file: Path,
    comments_file: Path,
    train_file: Path,
    dev_file: Path,
    eval_split: float,
    span_key: str,
):
    """Parse the ToxicSpan annotations into a training and development set for Spancat."""

    msg.info("Processing ToxicSpans")
    # Import data
    spans_df = pd.read_csv(spans_file)
    annotations_df = pd.read_csv(annotations_file)
    comments_df = pd.read_csv(comments_file)

    # Create spans_indices list
    spans_indices_list = []
    spans_labels = {}
    for index, row in spans_df.iterrows():
        if str(row["type"]) in ["Insult"]:
            label = str(row["type"]).upper().replace(" ", "_")
            if label not in spans_labels:
                spans_labels[label] = 0
            spans_labels[label] += 1
            spans_indices_list.append(
                (row["annotation"], label, [row["start"], row["end"]])
            )

    # Create annotations_list
    annotation_dict = {}
    for index, row in annotations_df.iterrows():
        if row["annotation"] not in annotation_dict:
            annotation_dict[row["annotation"]] = row["comment_id"]

    # Create comments_list
    comments_dict = {}
    for index, row in comments_df.iterrows():
        if row["comment_id"] not in comments_dict:
            comments_dict[row["comment_id"]] = row["comment_text"]

    # Initialize blank model
    nlp = spacy.blank("en")

    def _character_offset_to_token(doc: Doc, offsets: list) -> list:
        token_list = []
        for token in doc:
            if offsets[0] == token.idx:
                token_list.append(token.i)
            elif token.idx > offsets[0] and token.idx <= offsets[1]:
                token_list.append(token.i)
        return token_list

    doc_dict = {}
    for span in tqdm(spans_indices_list, total=len(spans_indices_list)):
        doc_id = annotation_dict[span[0]]

        if doc_id not in doc_dict:
            doc_dict[doc_id] = nlp(comments_dict[annotation_dict[span[0]]])
            doc_dict[doc_id].spans[span_key] = []

        doc = doc_dict[doc_id]
        tokens = _character_offset_to_token(doc, span[2])
        if not tokens:
            continue
        span_object = Span(doc, tokens[0], tokens[-1] + 1, span[1])
        doc.spans[span_key].append(span_object)

    # Split
    docs = list(doc_dict.values())
    train = []
    dev = []

    split = int(len(docs) * eval_split)
    train = docs[split:]
    dev = docs[:split]

    # Get max span length from train
    max_span_length = 0
    for doc in train:
        for span in doc.spans[span_key]:
            span_length = span.end - span.start
            if span_length > max_span_length:
                max_span_length = span_length

    # Save to disk
    docbin = DocBin(docs=train, store_user_data=True)
    docbin.to_disk(train_file)

    docbin = DocBin(docs=dev, store_user_data=True)
    docbin.to_disk(dev_file)
    msg.good(f"Processing ToxicSpans done")


if __name__ == "__main__":
    typer.run(main)
