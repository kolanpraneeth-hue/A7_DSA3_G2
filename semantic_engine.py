"""
Semantic similarity engine using Sentence-Transformers + FAISS.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Match:
    query_idx: int
    ref_idx: int
    score: float
    query_sentence: str
    ref_sentence: str


@dataclass
class SimilarityResult:
    overall_score: float          # 0–100 percentage
    decision: str                 # LOW / MEDIUM / HIGH
    matches: list[Match]
    query_count: int
    ref_count: int
    model_name: str
    threshold_high: float
    threshold_medium: float


class SemanticEngine:
    """
    Embed sentences, index reference with FAISS, compare query document.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold_high: float = 0.65,
        threshold_medium: float = 0.40,
        top_k: int = 3,
    ):
        self.model_name = model_name
        self.threshold_high = threshold_high
        self.threshold_medium = threshold_medium
        self.top_k = top_k

        print(f"[INFO] Loading Sentence-Transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()

        self.ref_sentences: list[str] = []
        self.index: faiss.IndexFlatIP | None = None  # inner-product on normalised vectors = cosine

    def _embed(self, sentences: list[str]) -> np.ndarray:
        """Return L2-normalised embeddings (float32)."""
        emb = self.model.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return emb.astype("float32")

    def build_index(self, ref_sentences: list[str]) -> None:
        """Create FAISS index from reference sentences."""
        if not ref_sentences:
            raise ValueError("Reference document has no usable sentences.")

        self.ref_sentences = ref_sentences
        print(f"[INFO] Reference sentences : {len(ref_sentences)}")
        print(f"[INFO] Building FAISS index (dim={self.dim}) ...")

        vectors = self._embed(ref_sentences)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)

    def compare(self, query_sentences: list[str]) -> SimilarityResult:
        """
        Compare query sentences against the indexed reference.
        Returns overall score, decision class and top sentence matches.
        """
        if self.index is None:
            raise RuntimeError("Call build_index() before compare().")

        if not query_sentences:
            raise ValueError("Query document has no usable sentences.")

        print(f"[INFO] Query sentences : {len(query_sentences)}")
        print("[INFO] Computing semantic nearest-neighbour matches ...")

        q_vectors = self._embed(query_sentences)
        k = min(self.top_k, len(self.ref_sentences))
        scores, indices = self.index.search(q_vectors, k)

        # Collect strong matches (best neighbour per query sentence)
        all_matches: list[Match] = []
        best_scores: list[float] = []

        for qi, (row_scores, row_idxs) in enumerate(zip(scores, indices)):
            best = float(row_scores[0])
            best_scores.append(best)
            for sc, ri in zip(row_scores, row_idxs):
                if ri < 0:
                    continue
                all_matches.append(
                    Match(
                        query_idx=qi,
                        ref_idx=int(ri),
                        score=float(sc),
                        query_sentence=query_sentences[qi],
                        ref_sentence=self.ref_sentences[int(ri)],
                    )
                )

        # Overall score = mean of each query sentence's best match
        overall = float(np.mean(best_scores)) if best_scores else 0.0
        percentage = round(overall * 100, 1)

        if overall >= self.threshold_high:
            decision = "HIGH"
        elif overall >= self.threshold_medium:
            decision = "MEDIUM"
        else:
            decision = "LOW"

        # Sort matches by score descending and keep unique strong pairs
        all_matches.sort(key=lambda m: m.score, reverse=True)
        seen = set()
        top_matches: list[Match] = []
        for m in all_matches:
            key = (m.query_idx, m.ref_idx)
            if key in seen:
                continue
            if m.score < 0.45:  # ignore weak matches in evidence list
                continue
            seen.add(key)
            top_matches.append(m)
            if len(top_matches) >= 8:
                break

        return SimilarityResult(
            overall_score=percentage,
            decision=decision,
            matches=top_matches,
            query_count=len(query_sentences),
            ref_count=len(self.ref_sentences),
            model_name=self.model_name,
            threshold_high=self.threshold_high,
            threshold_medium=self.threshold_medium,
        )
