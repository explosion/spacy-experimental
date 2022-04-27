import spacy
from spacy.tokens import DocBin
from pathlib import Path
from tqdm import tqdm
from wasabi import msg
import typer


def main(
    span_key: str,
    model_path: Path,
    test_path: Path,
):
    # Initialize NER & Spancat models
    nlp = spacy.load(model_path)
    spancat = nlp.get_pipe("spancat")

    # Get test.spacy DocBin
    test_doc_bin = DocBin().from_disk(test_path)
    test_docs = list(test_doc_bin.get_docs(nlp.vocab))

    # Suggester KPI
    total_candidates = 0
    total_real_candidates = 0
    matching_candidates = 0

    msg.info("Starting evaluation")

    for test_doc in tqdm(
        test_docs, total=len(test_docs), desc=f"Evaluation test dataset"
    ):
        # Prediction
        text = test_doc.text
        doc = nlp(text)
        spancat.set_candidates([doc])

        # Count spans when saving spans is enabled
        total_candidates += len(doc.spans["candidates"])
        total_real_candidates += len(test_doc.spans[span_key])

        # Check for True Positives and False Positives
        for test_span in test_doc.spans[span_key]:
            # Calculate coverage
            for span in doc.spans["candidates"]:
                if span.start == test_span.start and span.end == test_span.end:
                    matching_candidates += 1

    msg.good("Evaluation successful")

    # Suggester Coverage
    coverage = round((matching_candidates / total_real_candidates) * 100, 2)
    candidates_relation = round((total_candidates / total_real_candidates) * 100, 2)

    msg.divider("Suggester KPI")

    suggester_header = ["KPI", "Value"]
    suggester_data = [
        ("Suggester candidates", total_candidates),
        ("Real candidates", total_real_candidates),
        ("% Ratio", f"{candidates_relation}%"),
        ("% Coverage", f"{coverage}%"),
    ]
    msg.table(suggester_data, header=suggester_header, divider=True)


if __name__ == "__main__":
    typer.run(main)
