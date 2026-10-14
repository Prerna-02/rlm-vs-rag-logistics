"""Retrieval over records: BM25 (keywords), dense (embeddings) and hybrid (both, fused by rank).

Each record (one order line) is one retrievable chunk. Indexes are built once over the 2M context;
because smaller contexts are prefixes of it, search(..., n=N) simply ignores records past N.
"""
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

EMBED_MODEL = "intfloat/multilingual-e5-small"  # 118M params, multilingual (Portuguese reviews, English questions)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def top_k(scores: np.ndarray, k: int, n: int) -> list[int]:
    """Indices of the k best scores among the first n records."""
    return np.argsort(-scores[:n])[:k].tolist()


class BM25:
    def __init__(self, lines: list[str]):
        self.index = BM25Okapi([tokenize(l) for l in lines])

    def scores(self, query: str) -> np.ndarray:
        return self.index.get_scores(tokenize(query))

    def search(self, query: str, k: int, n: int) -> list[int]:
        return top_k(self.scores(query), k, n)


class Dense:
    def __init__(self, lines: list[str], cache: Path | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(EMBED_MODEL, device="cpu")
        if cache and cache.exists():
            self.emb = np.load(cache)
        else:  # e5 models expect "passage: " / "query: " prefixes
            self.emb = self.model.encode([f"passage: {l}" for l in lines], batch_size=64,
                                         normalize_embeddings=True, show_progress_bar=True)
            if cache:
                np.save(cache, self.emb)

    def scores(self, query: str) -> np.ndarray:
        q = self.model.encode(f"query: {query}", normalize_embeddings=True)
        return self.emb @ q  # cosine similarity, since both sides are normalized

    def search(self, query: str, k: int, n: int) -> list[int]:
        return top_k(self.scores(query), k, n)


class Hybrid:
    """Reciprocal rank fusion: each record scores 1/(c + rank) in each list; the sums decide.
    Ranks rather than raw scores are fused because BM25 and cosine scores live on different scales."""

    def __init__(self, bm25: BM25, dense: Dense, depth: int = 100, c: int = 60):
        self.parts, self.depth, self.c = (bm25, dense), depth, c

    def search(self, query: str, k: int, n: int) -> list[int]:
        fused: dict[int, float] = {}
        for part in self.parts:
            for rank, i in enumerate(part.search(query, self.depth, n)):
                fused[i] = fused.get(i, 0) + 1 / (self.c + rank)
        return sorted(fused, key=fused.get, reverse=True)[:k]
