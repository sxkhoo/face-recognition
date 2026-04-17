"""Watchlist matching — FAISS with numpy fallback."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.types import MatchResult

logger = logging.getLogger(__name__)

# Try FAISS; fall back to pure numpy
try:
    import faiss

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    logger.info("FAISS not available — using numpy cosine similarity fallback.")


class WatchlistManager:
    """Manages enrolled identities and performs nearest-neighbor matching.

    Embeddings are L2-normalized, so inner-product == cosine similarity
    and cosine distance = 1 - dot(a, b).
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        threshold: float = 0.4,
        use_faiss: bool = True,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.threshold = threshold
        self._use_faiss = use_faiss and _HAS_FAISS

        # Storage
        self._embeddings: List[np.ndarray] = []       # each (D,)
        self._identities: List[Dict] = []             # {id, name, ...}

        # FAISS index (rebuilt on changes)
        self._index: Optional[faiss.IndexFlatIP] = None if self._use_faiss else None

    @property
    def size(self) -> int:
        return len(self._identities)

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def remove(self, identity_id: str) -> int:
        """Remove all embeddings for an identity. Returns count removed."""
        indices = [i for i, ident in enumerate(self._identities) if ident["id"] == identity_id]
        if not indices:
            return 0
        for i in reversed(indices):
            self._embeddings.pop(i)
            self._identities.pop(i)
        self._rebuild_index()
        logger.info("Removed %d entries for id=%s — watchlist size: %d", len(indices), identity_id, self.size)
        return len(indices)

    def enroll(
        self,
        identity_id: str,
        name: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add a person to the watchlist.

        If the identity already exists, the embedding is appended
        (multiple templates per person are supported).
        """
        emb = self._normalize(embedding)
        self._embeddings.append(emb)
        entry = {"id": identity_id, "name": name, **(metadata or {})}
        self._identities.append(entry)
        self._rebuild_index()
        logger.info("Enrolled %s (%s) — watchlist size: %d", name, identity_id, self.size)

    def enroll_batch(
        self,
        identity_id: str,
        name: str,
        embeddings: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Enroll multiple embeddings for one identity."""
        for emb in embeddings:
            self._embeddings.append(self._normalize(emb))
            self._identities.append({"id": identity_id, "name": name, **(metadata or {})})
        self._rebuild_index()

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(self, embedding: np.ndarray) -> MatchResult:
        """Find best watchlist match for a query embedding.

        Returns MatchResult with matched=False if watchlist empty or no
        match above threshold.
        """
        if self.size == 0:
            return MatchResult(
                identity_id="",
                name="unknown",
                distance=1.0,
                matched=False,
            )

        query = self._normalize(embedding).reshape(1, -1).astype(np.float32)

        if self._use_faiss and self._index is not None:
            similarities, indices = self._index.search(query, 1)
            sim = float(similarities[0, 0])
            idx = int(indices[0, 0])
        else:
            gallery = np.stack(self._embeddings).astype(np.float32)
            sims = gallery @ query.T  # (N, 1)
            idx = int(np.argmax(sims))
            sim = float(sims[idx, 0])

        distance = 1.0 - sim
        identity = self._identities[idx]
        matched = distance < self.threshold

        logger.debug(
            "Match query: best=%s (id=%s), dist=%.4f, thresh=%.4f, matched=%s",
            identity["name"],
            identity["id"],
            distance,
            self.threshold,
            matched,
        )

        return MatchResult(
            identity_id=identity["id"],
            name=identity["name"],
            distance=distance,
            matched=matched,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, embeddings_path: str, identities_path: str) -> None:
        """Save watchlist to disk."""
        if self.size == 0:
            logger.warning("Watchlist empty — nothing to save.")
            return
        emb_array = np.stack(self._embeddings)
        np.save(embeddings_path, emb_array)
        with open(identities_path, "w") as f:
            json.dump(self._identities, f, indent=2)
        logger.info("Saved watchlist: %d entries to %s", self.size, embeddings_path)

    def load(self, embeddings_path: str, identities_path: str) -> None:
        """Load watchlist from disk."""
        emb_path = Path(embeddings_path)
        id_path = Path(identities_path)
        if not emb_path.exists() or not id_path.exists():
            logger.warning("Watchlist files not found — starting empty.")
            return
        emb_array = np.load(str(emb_path))
        with open(str(id_path)) as f:
            identities = json.load(f)
        if emb_array.shape[0] != len(identities):
            raise ValueError(
                f"Watchlist mismatch: {emb_array.shape[0]} embeddings vs "
                f"{len(identities)} identities in {embeddings_path} / {identities_path}"
            )
        self._identities = identities
        self._embeddings = [emb_array[i] for i in range(emb_array.shape[0])]
        self._rebuild_index()
        logger.info("Loaded watchlist: %d entries from %s", self.size, embeddings_path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float32).flatten()
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v

    def _rebuild_index(self) -> None:
        if not self._use_faiss or not _HAS_FAISS:
            return
        if self.size == 0:
            self._index = None
            return
        matrix = np.stack(self._embeddings).astype(np.float32)
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        self._index.add(matrix)
