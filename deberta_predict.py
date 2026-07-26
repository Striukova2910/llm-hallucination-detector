import argparse

from src.config import NLI_MODEL_NAME
from src.nli_inference import NliHallucinationDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether an answer is supported, unverifiable, "
            "or contradicted by reference evidence."
        )
    )
    parser.add_argument(
        "--model",
        default=NLI_MODEL_NAME,
        help="Hugging Face model name or a local model path.",
    )
    parser.add_argument("--question")
    parser.add_argument("--answer")
    parser.add_argument("--evidence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    question = args.question or input("Question: ").strip()
    answer = args.answer or input("Answer to check: ").strip()
    evidence = args.evidence or input("Reference evidence: ").strip()

    if not question:
        raise ValueError("Question must be non-empty.")

    if not answer:
        raise ValueError("Answer must be non-empty.")

    if not evidence:
        raise ValueError("Reference evidence must be non-empty.")

    detector = NliHallucinationDetector(args.model)
    prediction = detector.predict(evidence, answer)

    print(f"\nStatus: {prediction.status}")
    print(
        "Entailment probability: "
        f"{prediction.entailment_probability:.1%}"
    )
    print(
        "Neutral probability: "
        f"{prediction.neutral_probability:.1%}"
    )
    print(
        "Contradiction probability: "
        f"{prediction.contradiction_probability:.1%}"
    )
    print(
        "Groundedness risk: "
        f"{prediction.groundedness_risk:.1%}"
    )


if __name__ == "__main__":
    main()
