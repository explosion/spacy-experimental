from pathlib import Path
from typing import List

import typer
from spacy.tokens import Doc, DocBin, SpanGroup
from spacy.training.converters import conll_ner_to_docs
from wasabi import msg

DOC_DELIMITER = "-DOCSTART- -X- O O\n"


def parse_genia(
    data: str, num_levels: int = 4, doc_delimiter: str = DOC_DELIMITER
) -> List[Doc]:
    """Parse GENIA dataset into spaCy docs

    Our strategy here is to reuse the conll -> ner method from
    spaCy and re-apply that n times. We don't want to write our
    own ConLL/IOB parser.

    Parameters
    ----------
    data: str
        The raw string input as read from the IOB file
    num_levels: int, default is 4
        Represents how many times a label has been nested. In
        GENIA, a label was nested four times at maximum.

    Returns
    -------
    List[Doc]
    """
    docs = data.split("\n\n")
    iob_per_level = []
    for level in range(num_levels):
        doc_list = []
        for doc in docs:
            tokens = [t for t in doc.split("\n") if t]
            token_list = []
            for token in tokens:
                annot = token.split("\t")
                # First element is always the token text
                text = annot[0]
                label = annot[level + 1]
                _token = " ".join([text, label])
                token_list.append(_token)
            doc_list.append("\n".join(token_list))
        annotations = doc_delimiter.join(doc_list)
        iob_per_level.append(annotations)

    # We then copy all the entities from doc.ents into
    # doc.spans later on. But first, let's have a "canonical" docs
    # to copy into
    docs_per_level = [list(conll_ner_to_docs(iob)) for iob in iob_per_level]
    docs_with_spans: List[Doc] = []

    for docs in zip(*docs_per_level):
        spans = [ent for doc in docs for ent in doc.ents]
        doc = docs[0]
        group = SpanGroup(doc, name="sc", spans=spans)
        doc.spans["sc"] = group
        docs_with_spans.append(doc)

    return docs_with_spans


def main(
    train_path: Path, dev_path: Path, train_output_path: Path, dev_output_path: Path
):

    msg.good(f"Processing Genia")
    with train_path.open("r", encoding="utf-8") as f:
        train_data = f.read()
    with dev_path.open("r", encoding="utf-8") as f:
        dev_data = f.read()

    train_docs = parse_genia(train_data)
    dev_docs = parse_genia(dev_data)

    train_doc_bin = DocBin(docs=train_docs)
    dev_doc_bin = DocBin(docs=dev_docs)

    train_doc_bin.to_disk(train_output_path)
    dev_doc_bin.to_disk(dev_output_path)

    msg.good(f"Processing Genia done")


if __name__ == "__main__":
    typer.run(main)
