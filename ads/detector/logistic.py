"""Learned groundedness detector based on logistic regression."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ads.detector.threshold import DetectorOutput
from ads.features.density import DensityFeatures

FEATURE_COLUMNS = [
    "entropy_top_k",
    "top1_share",
    "top5_share",
    "peakiness_ratio",
    "gini",
    "max_score",
    "effective_k",
]


def encode_label(label: str | int | bool) -> int:
    """Encode user-facing labels into binary {0,1} targets."""
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, int):
        return int(label > 0)
    normalized = label.strip().lower()
    return int(normalized in {"faithful", "grounded", "1", "true", "yes"})


class LogisticDetector:
    """Wrapper around StandardScaler + LogisticRegression with model IO."""

    def __init__(
        self,
        random_state: int = 42,
        feature_columns: Iterable[str] = FEATURE_COLUMNS,
    ) -> None:
        """Initialize model state and preprocessing pipeline."""
        self.random_state = random_state
        self.feature_columns = list(feature_columns)
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=2000, random_state=random_state)
        self._fitted = False

    def fit(self, frame: pd.DataFrame, label_column: str = "label") -> LogisticDetector:
        """Fit detector from a labeled feature table."""
        y = frame[label_column].apply(encode_label).to_numpy(dtype=int)
        x = frame[self.feature_columns].to_numpy(dtype=float)
        x_scaled = self.scaler.fit_transform(x)
        self.model.fit(x_scaled, y)
        self._fitted = True
        return self

    def predict_score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict groundedness probabilities for each row."""
        self._assert_fitted()
        x = frame[self.feature_columns].to_numpy(dtype=float)
        x_scaled = self.scaler.transform(x)
        probabilities: np.ndarray = self.model.predict_proba(x_scaled)[:, 1]
        return probabilities.astype(float)

    def predict_output(
        self,
        features: DensityFeatures,
        threshold: float = 0.5,
    ) -> DetectorOutput:
        """Predict detector output for a single feature bundle."""
        self._assert_fitted()
        frame = pd.DataFrame([features.to_dict()])
        score = float(self.predict_score_frame(frame)[0])
        predicted_label = int((score >= threshold) and not features.abstain_flag)
        confidence = float(max(score, 1.0 - score))
        return DetectorOutput(
            groundedness_score=score,
            predicted_label=predicted_label,
            confidence=confidence,
            abstain_flag=features.abstain_flag,
        )

    def save(self, path: str | Path) -> None:
        """Persist trained detector to disk."""
        self._assert_fitted()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "random_state": self.random_state,
            "feature_columns": self.feature_columns,
            "scaler": self.scaler,
            "model": self.model,
        }
        joblib.dump(payload, output_path)

    @classmethod
    def load(cls, path: str | Path) -> LogisticDetector:
        """Load persisted detector from disk."""
        payload = joblib.load(Path(path))
        feature_columns = payload.get("feature_columns", FEATURE_COLUMNS)
        detector = cls(
            random_state=int(payload["random_state"]),
            feature_columns=list(feature_columns),
        )
        detector.scaler = payload["scaler"]
        detector.model = payload["model"]
        detector._fitted = True
        return detector

    def _assert_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("LogisticDetector must be fitted before prediction or save.")
