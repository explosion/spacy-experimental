from numpy import mat
import spacy
from spacy.tokens import DocBin
from spacy.scorer import PRFScore
from pathlib import Path

from tqdm import tqdm
from wasabi import msg
from wasabi import table
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
    test_docBin = DocBin().from_disk(test_path)
    test_docs = list(test_docBin.get_docs(spacy.blank("en").vocab))

    # Initialize scorers
    scorer = {}
    for label in spancat.labels:
        scorer[label] = PRFScore()

    KPI = {}
    for label in scorer:
        KPI[label] = {
            "total_spans": 0,
            "correct_spans": 0,
        }

    doc_eval_list = []

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

        doc_eval = {
            "doc": test_doc.text,
            "false_positive": [],
            "true_positive": [],
            "false_negative": [],
        }

        # Check for True Positives and False Positives
        for test_span in test_doc.spans[span_key]:
            KPI[test_span.label_]["total_spans"] += 1
            match = False
            for span in doc.spans[span_key]:
                if span.start == test_span.start and span.end == test_span.end:
                    if span.label_ == test_span.label_:
                        scorer[test_span.label_].tp += 1
                        doc_eval["true_positive"].append(test_span)
                        match = True
                        break
                    else:
                        scorer[span.label_].fp += 1
                        doc_eval["false_positive"].append(span)

            # Calculate coverage
            for span in doc.spans["candidates"]:
                if span.start == test_span.start and span.end == test_span.end:
                    matching_candidates += 1

            if not match:
                scorer[test_span.label_].fn += 1
                doc_eval["false_negative"].append(test_span)

        # Check for False Positives from the Spancat
        for span in doc.spans[span_key]:
            match = False
            for test_span in test_doc.spans[span_key]:
                if span.start == test_span.start and span.end == test_span.end:
                    match = True
                    break
            if not match:
                scorer[span.label_].fp += 1
                doc_eval["false_positive"].append(span)

        doc_eval_list.append(doc_eval)

    msg.good("Evaluation successful")

    # Table Config
    header = ("Label", "F-Score", "Recall", "Precision")

    # Spancat Table
    spancat_data = []
    spancat_fscore = 0
    spancat_recall = 0
    spancat_precision = 0

    for label in scorer:
        spancat_data.append(
            (
                label,
                round(scorer[label].fscore, 2),
                round(scorer[label].recall, 2),
                round(scorer[label].precision, 2),
            )
        )
        spancat_fscore += scorer[label].fscore
        spancat_recall += scorer[label].recall
        spancat_precision += scorer[label].precision

    spancat_fscore /= len(scorer)
    spancat_recall /= len(scorer)
    spancat_precision /= len(scorer)

    spancat_data.append(
        (
            "Average",
            round(spancat_fscore, 2),
            round(spancat_recall, 2),
            round(spancat_precision, 2),
        )
    )

    msg.divider("Spancat")
    print(table(spancat_data, header=header, divider=True))

    print()

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
        ("F-Score", round(spancat_fscore,2)),
        ("Recall", round(spancat_recall,2)),
        ("Precision", round(spancat_precision,2)),
    ]
    print(table(suggester_data, header=suggester_header, divider=True))

if __name__ == "__main__":
    typer.run(main)
