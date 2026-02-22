"""Rule-based threshold groundedness detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ads.features.density import DensityFeatures


@dataclass(slots=True)
class DetectorOutput:
    """Unified detector prediction output."""

    groundedness_score: float
    predicted_label: int
    confidence: float
    abstain_flag: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        """Convert to dict for JSON output."""
        return asdict(self)


class ThresholdDetector:
    """Detector using concentration-oriented density heuristics."""

    def __init__(self, score_threshold: float = 0.55, score_floor: float = 0.05) -> None:
        """Initialize rule-based thresholds for groundedness decisions."""
        self.score_threshold = score_threshold
        self.score_floor = score_floor

    def score(self, features: DensityFeatures) -> float:
        """Compute groundedness score from feature bundle."""
        concentration = 1.0 - features.entropy_top_k
        score = 0.5 * features.top1_share + 0.35 * features.top5_share + 0.15 * concentration
        if features.max_score < self.score_floor:
            score *= 0.5
        return float(max(0.0, min(1.0, score)))

    def predict(self, features: DensityFeatures) -> DetectorOutput:
        """Return label, confidence, and abstain decision."""
        groundedness_score = self.score(features)
        abstain_flag = features.abstain_flag or features.max_score < self.score_floor
        predicted_label = int((groundedness_score >= self.score_threshold) and not abstain_flag)
        confidence = float(min(1.0, abs(groundedness_score - self.score_threshold) * 2.0))
        return DetectorOutput(
            groundedness_score=groundedness_score,
            predicted_label=predicted_label,
            confidence=confidence,
            abstain_flag=abstain_flag,
        )
