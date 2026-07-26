import pytest

from src.nli_inference import NliPrediction, select_nli_status


@pytest.mark.parametrize(
    ("probabilities", "expected_status"),
    [
        ((0.90, 0.05, 0.05), "supported"),
        ((0.05, 0.90, 0.05), "unverifiable"),
        ((0.05, 0.05, 0.90), "contradicted"),
    ],
)
def test_select_nli_status(probabilities, expected_status):
    assert select_nli_status(*probabilities) == expected_status


def test_prediction_exposes_groundedness_risk():
    prediction = NliPrediction(
        status="unverifiable",
        entailment_probability=0.10,
        neutral_probability=0.70,
        contradiction_probability=0.20,
    )

    assert prediction.groundedness_risk == pytest.approx(0.90)
    assert prediction.is_direct_hallucination is False
