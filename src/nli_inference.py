"""Evidence-grounded hallucination detection with a pretrained NLI model."""

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import NLI_MAX_LENGTH, NLI_MODEL_NAME


@dataclass(frozen=True)
class NliPrediction:
    """Three-way NLI prediction for an answer and its reference evidence."""

    status: str
    entailment_probability: float
    neutral_probability: float
    contradiction_probability: float

    @property
    def groundedness_risk(self) -> float:
        """Probability that the answer is not fully supported by the evidence."""
        return self.neutral_probability + self.contradiction_probability

    @property
    def is_direct_hallucination(self) -> bool:
        """Whether the most probable NLI class is a direct contradiction."""
        return self.status == "contradicted"


def select_nli_status(
    entailment_probability: float,
    neutral_probability: float,
    contradiction_probability: float,
) -> str:
    """Map the largest NLI probability to a user-facing status."""
    probabilities = (
        entailment_probability,
        neutral_probability,
        contradiction_probability,
    )
    statuses = ("supported", "unverifiable", "contradicted")
    return statuses[max(range(len(probabilities)), key=probabilities.__getitem__)]


def default_device() -> torch.device:
    """Select CUDA, Apple Silicon MPS, or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


class NliHallucinationDetector:
    """Compare an answer with reference evidence using three-way NLI."""

    def __init__(
        self,
        model_name_or_path: str | Path = NLI_MODEL_NAME,
        device: str | torch.device | None = None,
    ):
        self.model_name_or_path = str(model_name_or_path)
        self.device = torch.device(device) if device else default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path
        ).to(self.device)
        self.model.eval()

        label_to_id = {
            str(label).lower(): int(label_id)
            for label_id, label in self.model.config.id2label.items()
        }
        required_labels = {"entailment", "neutral", "contradiction"}
        missing_labels = required_labels.difference(label_to_id)
        if missing_labels:
            raise ValueError(
                "The selected model is not a compatible three-way NLI model. "
                f"Missing labels: {sorted(missing_labels)}"
            )

        self.entailment_id = label_to_id["entailment"]
        self.neutral_id = label_to_id["neutral"]
        self.contradiction_id = label_to_id["contradiction"]

    def predict(self, reference_evidence: str, answer: str) -> NliPrediction:
        """Return support, uncertainty, and contradiction probabilities."""
        if not reference_evidence.strip() or not answer.strip():
            raise ValueError(
                "Reference evidence and answer must both be non-empty."
            )

        encoded = self.tokenizer(
            reference_evidence,
            answer,
            return_tensors="pt",
            truncation=True,
            max_length=NLI_MAX_LENGTH,
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[0].cpu()

        entailment = float(probabilities[self.entailment_id])
        neutral = float(probabilities[self.neutral_id])
        contradiction = float(probabilities[self.contradiction_id])

        return NliPrediction(
            status=select_nli_status(entailment, neutral, contradiction),
            entailment_probability=entailment,
            neutral_probability=neutral,
            contradiction_probability=contradiction,
        )
