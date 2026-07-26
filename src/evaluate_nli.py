"""Evaluate DeBERTa NLI on an external counterfactual stress test."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import DATA_DIR, NLI_MODEL_NAME, REPORTS_DIR
from src.nli_inference import NliHallucinationDetector


REQUIRED_COLUMNS = {
    "reference_evidence",
    "answer",
    "expected_label",
    "answer_style",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the pretrained NLI detector on a CSV stress test."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR / "counterfactual_test.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "deberta_ood_results.csv",
    )
    parser.add_argument("--model", default=NLI_MODEL_NAME)
    return parser.parse_args()


def binary_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_probability: pd.Series,
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probability),
        "pr_auc": average_precision_score(y_true, y_probability),
    }


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}")
    for name, value in metrics.items():
        print(f"{name:>10}: {value:.4f}")


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Stress-test data not found: {args.data}")

    frame = pd.read_csv(args.data)
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing_columns)}"
        )

    print(f"Loaded {len(frame)} external stress-test examples.")
    print(f"Loading NLI model: {args.model}")
    detector = NliHallucinationDetector(args.model)

    records: list[dict[str, object]] = []
    for row_number, row in frame.iterrows():
        prediction = detector.predict(
            row["reference_evidence"],
            row["answer"],
        )
        records.append(
            {
                "nli_status": prediction.status,
                "entailment_probability": prediction.entailment_probability,
                "neutral_probability": prediction.neutral_probability,
                "contradiction_probability": (
                    prediction.contradiction_probability
                ),
                "groundedness_risk": prediction.groundedness_risk,
                "strict_prediction": int(prediction.status != "supported"),
                "contradiction_prediction": int(
                    prediction.is_direct_hallucination
                ),
            }
        )
        print(
            f"\rEvaluated {row_number + 1}/{len(frame)} examples",
            end="",
            flush=True,
        )
    print()

    result = pd.concat(
        [frame.reset_index(drop=True), pd.DataFrame(records)],
        axis=1,
    )
    y_true = result["expected_label"].astype(int)

    strict_metrics = binary_metrics(
        y_true,
        result["strict_prediction"],
        result["groundedness_risk"],
    )
    contradiction_metrics = binary_metrics(
        y_true,
        result["contradiction_prediction"],
        result["contradiction_probability"],
    )
    print_metrics("Strict groundedness evaluation", strict_metrics)
    print_metrics("Contradiction-focused evaluation", contradiction_metrics)

    print("\nStrict confusion matrix [[TN, FP], [FN, TP]]")
    print(confusion_matrix(y_true, result["strict_prediction"], labels=[0, 1]))
    print("\nContradiction confusion matrix [[TN, FP], [FN, TP]]")
    print(
        confusion_matrix(
            y_true,
            result["contradiction_prediction"],
            labels=[0, 1],
        )
    )

    print("\nNLI status by expected label")
    print(pd.crosstab(result["expected_label"], result["nli_status"]))
    print("\nNLI status by answer style")
    print(pd.crosstab(result["answer_style"], result["nli_status"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"\nDetailed predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
