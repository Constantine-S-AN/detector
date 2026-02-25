"""DDA backends: TF-IDF proxy and optional real reproduction backend."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ads.attribution.base import AttributionBackend, AttributionItem
from ads.io import read_jsonl


class DDATfidfProxyBackend(AttributionBackend):
    """Lightweight DDA proxy using TF-IDF similarity + debiasing.

    Score semantics: larger score => more influential training sample.
    influence = max(0, sim(answer, train_i) - alpha * sim(prompt, train_i))
    """

    name = "dda_tfidf_proxy"

    def __init__(
        self,
        train_corpus_path: str | Path,
        seed: int = 42,
        *,
        alpha: float = 0.35,
        min_score: float = 0.0,
        cache_dir: str | Path = ".cache/ads/dda_tfidf_proxy",
        model_id: str = "dda_tfidf_proxy_v1",
    ) -> None:
        self._train_corpus_path = Path(train_corpus_path)
        self._seed = seed
        self._alpha = float(alpha)
        self._min_score = float(min_score)
        self._cache_dir = Path(cache_dir)
        self._model_id = model_id

        if not self._train_corpus_path.exists():
            raise FileNotFoundError(f"train corpus not found: {self._train_corpus_path}")

        rows = read_jsonl(self._train_corpus_path)
        if not rows:
            raise ValueError("DDATfidfProxyBackend requires non-empty train corpus")

        self._train_rows = rows
        self._train_texts = [str(row.get("text", "")) for row in rows]
        self._train_ids = [str(row.get("train_id", idx)) for idx, row in enumerate(rows)]

        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._train_matrix = self._vectorizer.fit_transform(self._train_texts)
        self._train_corpus_hash = self._hash_file(self._train_corpus_path)

    def compute(
        self,
        prompt: str,
        answer: str,
        top_k: int,
        *,
        sample_meta: dict[str, Any] | None = None,
        attribution_mode: str | None = None,
    ) -> list[AttributionItem]:
        """Compute top-k influence list with stable cache."""
        del attribution_mode
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        sample_id = ""
        if sample_meta is not None and sample_meta.get("sample_id") is not None:
            sample_id = str(sample_meta["sample_id"])

        cache_path = self._cache_path(
            prompt=prompt, answer=answer, top_k=top_k, sample_id=sample_id
        )
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return [
                AttributionItem(
                    train_id=str(item["train_id"]),
                    score=float(item["score"]),
                    rank=int(item["rank"]),
                    text=str(item.get("text", "")),
                    source=str(item.get("source", "dda_tfidf_proxy")),
                    meta=dict(item.get("meta", {})),
                )
                for item in payload.get("items", [])
            ]

        query_matrix = self._vectorizer.transform([prompt, answer])
        prompt_vec = query_matrix[0]
        answer_vec = query_matrix[1]

        prompt_sim = (self._train_matrix @ prompt_vec.T).toarray().ravel()
        answer_sim = (self._train_matrix @ answer_vec.T).toarray().ravel()

        influence = answer_sim - self._alpha * prompt_sim
        influence = np.maximum(influence, self._min_score)

        items = _rank_to_items(
            influence=influence,
            train_rows=self._train_rows,
            train_ids=self._train_ids,
            source_default="dda_tfidf_proxy",
            extra_meta_builder=lambda idx: {
                "prompt_similarity": float(prompt_sim[idx]),
                "answer_similarity": float(answer_sim[idx]),
                "alpha": self._alpha,
                "model_id": self._model_id,
            },
            top_k=top_k,
        )

        payload = {
            "backend": self.name,
            "model_id": self._model_id,
            "train_corpus_hash": self._train_corpus_hash,
            "sample_id": sample_id,
            "k_requested": int(top_k),
            "k_effective": len(items),
            "params": {"alpha": self._alpha, "min_score": self._min_score},
            "items": [item.to_dict() for item in items],
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return items

    def _cache_path(self, *, prompt: str, answer: str, top_k: int, sample_id: str) -> Path:
        return _stable_cache_path(
            cache_dir=self._cache_dir,
            model_id=self._model_id,
            train_corpus_hash=self._train_corpus_hash,
            prompt=prompt,
            answer=answer,
            sample_id=sample_id,
            k=top_k,
            params={"alpha": self._alpha, "min_score": self._min_score, "backend": self.name},
        )

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()


class DDARealBackend(AttributionBackend):
    """Optional heavy-dependency DDA reproduction backend.

    Uses sentence-transformers embeddings as a practical reproducible approximation:
      influence = max(0, cos(answer, train_i) - alpha * cos(prompt, train_i))
    Score semantics: larger score => more influential.
    """

    name = "dda_real"

    def __init__(
        self,
        train_corpus_path: str | Path,
        seed: int = 42,
        *,
        alpha: float = 0.35,
        min_score: float = 0.0,
        cache_dir: str | Path = ".cache/ads/dda_real",
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        ckpt: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self._train_corpus_path = Path(train_corpus_path)
        self._seed = seed
        self._alpha = float(alpha)
        self._min_score = float(min_score)
        self._cache_dir = Path(cache_dir)
        self._model_id = model_id
        self._ckpt = str(ckpt) if ckpt is not None else None
        self._device = device

        if not self._train_corpus_path.exists():
            raise FileNotFoundError(f"train corpus not found: {self._train_corpus_path}")

        rows = read_jsonl(self._train_corpus_path)
        if not rows:
            raise ValueError("DDARealBackend requires non-empty train corpus")
        self._train_rows = rows
        self._train_texts = [str(row.get("text", "")) for row in rows]
        self._train_ids = [str(row.get("train_id", idx)) for idx, row in enumerate(rows)]
        self._train_corpus_hash = DDATfidfProxyBackend._hash_file(self._train_corpus_path)

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "DDA real backend dependencies are missing. Install with `pip install -e .[dda]` "
                "(or at least `pip install sentence-transformers torch`)."
            ) from exc

        model_name = self._ckpt or self._model_id
        self._encoder = SentenceTransformer(model_name, device=self._device)
        self._train_embeddings = self._encode(self._train_texts)

    def compute(
        self,
        prompt: str,
        answer: str,
        top_k: int,
        *,
        sample_meta: dict[str, Any] | None = None,
        attribution_mode: str | None = None,
    ) -> list[AttributionItem]:
        del attribution_mode
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        sample_id = ""
        if sample_meta is not None and sample_meta.get("sample_id") is not None:
            sample_id = str(sample_meta["sample_id"])

        cache_path = _stable_cache_path(
            cache_dir=self._cache_dir,
            model_id=self._model_id,
            train_corpus_hash=self._train_corpus_hash,
            prompt=prompt,
            answer=answer,
            sample_id=sample_id,
            k=top_k,
            params={
                "alpha": self._alpha,
                "min_score": self._min_score,
                "backend": self.name,
                "ckpt": self._ckpt,
                "device": self._device,
            },
        )
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return [
                AttributionItem(
                    train_id=str(item["train_id"]),
                    score=float(item["score"]),
                    rank=int(item["rank"]),
                    text=str(item.get("text", "")),
                    source=str(item.get("source", "dda_real")),
                    meta=dict(item.get("meta", {})),
                )
                for item in payload.get("items", [])
            ]

        prompt_emb, answer_emb = self._encode([prompt, answer])
        prompt_sim = self._cosine_with_train(prompt_emb)
        answer_sim = self._cosine_with_train(answer_emb)
        influence = np.maximum(answer_sim - self._alpha * prompt_sim, self._min_score)

        items = _rank_to_items(
            influence=influence,
            train_rows=self._train_rows,
            train_ids=self._train_ids,
            source_default="dda_real",
            extra_meta_builder=lambda idx: {
                "prompt_similarity": float(prompt_sim[idx]),
                "answer_similarity": float(answer_sim[idx]),
                "alpha": self._alpha,
                "model_id": self._model_id,
            },
            top_k=top_k,
        )

        payload = {
            "backend": self.name,
            "model_id": self._model_id,
            "ckpt": self._ckpt,
            "device": self._device,
            "train_corpus_hash": self._train_corpus_hash,
            "sample_id": sample_id,
            "k_requested": int(top_k),
            "k_effective": len(items),
            "params": {"alpha": self._alpha, "min_score": self._min_score},
            "items": [item.to_dict() for item in items],
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return items

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self._encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float64)

    def _cosine_with_train(self, query_emb: np.ndarray) -> np.ndarray:
        return np.matmul(self._train_embeddings, query_emb)


class DDABackend(DDATfidfProxyBackend):
    """Deprecated alias for backward compatibility with previous `dda` backend name."""

    name = "dda"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "`backend=dda` is deprecated and now maps to `dda_tfidf_proxy`. "
            "Use `dda_tfidf_proxy` or `dda_real` explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def _rank_to_items(
    *,
    influence: np.ndarray,
    train_rows: list[dict[str, Any]],
    train_ids: list[str],
    source_default: str,
    extra_meta_builder: Any,
    top_k: int,
) -> list[AttributionItem]:
    k_effective = min(int(top_k), len(train_rows))
    ranked_indices = np.argsort(influence)[::-1][:k_effective]
    items: list[AttributionItem] = []
    for rank, idx in enumerate(ranked_indices.tolist(), start=1):
        row = train_rows[idx]
        items.append(
            AttributionItem(
                train_id=train_ids[idx],
                score=float(influence[idx]),
                rank=rank,
                text=str(row.get("text", "")),
                source=str(row.get("source", source_default)),
                meta=extra_meta_builder(idx),
            )
        )
    return items


def _stable_cache_path(
    *,
    cache_dir: Path,
    model_id: str,
    train_corpus_hash: str,
    prompt: str,
    answer: str,
    sample_id: str,
    k: int,
    params: dict[str, Any],
) -> Path:
    sample_payload = {"prompt": prompt, "answer": answer, "sample_id": sample_id}
    sample_hash = hashlib.sha256(
        json.dumps(sample_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]

    key_payload = {
        "model_id": model_id,
        "train_corpus_hash": train_corpus_hash,
        "sample_hash": sample_hash,
        "k": int(k),
        "params": params,
    }
    cache_key = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return cache_dir / f"{cache_key}.json"
