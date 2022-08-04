import os
import glob

import spacy
import srsly
import typer
import tqdm

from collections import Counter
from pathlib import Path
from typing import Dict, Optional
from typing import List

from spacy.attrs import intify_attr
from spacy.tokens import DocBin


app = typer.Typer()


@app.command()
def make_mapper(
        path: Path,
        out_path: Path,
        *,
        attrs: Optional[List[str]] = ["NORM", "PREFIX", "SUFFIX", "SHAPE"],
        model: Optional[str] = None,
        language: Optional[str] = None,
        unk: int = 0,
        limit: Optional[int] = 0,
        min_freqs: Optional[List[int]] = [10, 10, 10, 10],
        max_symbols: Optional[List[int]] = [],
) -> None:
    error_msg = "One of 'model' or 'langauge' has to be provided"
    if min_freqs and len(min_freqs) != len(attrs):
        raise ValueError(
            "Have to provide same number of attrs and min_freqs."
        )
    if max_symbols and len(max_symbols) != len(attrs):
        raise ValueError(
            "Have to provide same number of attrs and max_symbols"
        )
    if model is None and language is None:
        raise ValueError(error_msg)
    elif model is not None and language is not None:
        raise ValueError(error_msg)
    else:
        if model:
            nlp = spacy.load(model)
        else:
            nlp = spacy.blank(language)
    attrs_counts = {}
    docbin = DocBin().from_disk(path)
    for attr in attrs:
        attr_id = intify_attr(attr)
        counts = Counter()
        for doc in tqdm.tqdm(
            docbin.get_docs(nlp.vocab), total=len(docbin)
        ):
            counts.update(doc.count_by(attr_id))
        attrs_counts[attr] = counts
    debug_path = os.path.join("debug", os.path.basename(path) + ".counts")
    srsly.write_msgpack(debug_path, attrs_counts)
    # Create mappers
    mappers: Dict[str, Dict[int, int]] = {}
    for i, attr in enumerate(attrs):
        sorted_counts = attrs_counts[attr].most_common()
        mappers[attr] = {}
        new_id = 0
        for j, (symbol, count) in enumerate(sorted_counts):
            if j == limit and limit != 0:
                break
            if min_freqs:
                if count < min_freqs[i]:
                    break
            if max_symbols:
                if len(mappers[attr]) > max_symbols[i]:
                    break
            # Leave the id for the unknown symbol out of the mapper.
            if new_id == unk:
                new_id += 1
            mappers[attr][symbol] = new_id
            new_id += 1
    srsly.write_msgpack(out_path, mappers)


if __name__ == "__main__":
    app()
