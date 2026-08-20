"""Query normalization and conservative semantic deduplication.

Two queries are considered duplicates only when they normalize to the same
canonical form or share a high token overlap (Jaccard >= threshold), target
the same provider and operation, and contain enough tokens to be meaningful.
Dedup is job-scoped: the persistent cache already handles cross-job reuse.
"""

from __future__ import annotations

import re
import threading
from typing import Dict, Optional, Set

_STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "or", "who", "what", "was", "were",
    "is", "are", "be", "to", "in", "on", "at", "about", "tell", "me", "explain",
    "give", "some", "information", "details", "detail", "background", "history",
    "biography", "bio", "overview", "please", "can", "you", "could", "would",
    "need", "want", "get", "find", "show", "describe", "summary", "summarize",
    "stuff", "things", "thing", "related", "concerning", "regarding",
}

_WORD_PATTERN = re.compile(r"[a-z0-9]{2,}")


def normalize_query(query: str) -> str:
    """Lowercase, strip punctuation, drop stopwords, collapse whitespace."""
    if not query:
        return ""
    lowered = query.lower()
    words = [
        word
        for word in _WORD_PATTERN.findall(lowered)
        if word not in _STOPWORDS
    ]
    return " ".join(words)


def token_set(query: str) -> Set[str]:
    return set(normalize_query(query).split())


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    intersection = left_tokens & right_tokens
    return len(intersection) / len(union)


class JobDedupStore:
    """Job-scoped dedup table mapping canonical keys to results.

    Thread-safe; reuse only happens when the canonical key matches exactly or
    token similarity is high (>= 0.75) and both queries have >= 3 tokens.
    """

    def __init__(self, threshold: float = 0.75, min_tokens: int = 3) -> None:
        self.threshold = threshold
        self.min_tokens = min_tokens
        self._lock = threading.Lock()
        self._store: Dict[str, Dict] = {}
        self._keys: Dict[str, str] = {}  # provider|operation|canonical -> payload key

    def find(
        self, provider: str, operation: str, query: str
    ) -> Optional[Dict]:
        """Return the stored payload when ``query`` duplicates an earlier one."""
        canonical = normalize_query(query)
        if not canonical:
            return None
        payload_key = f"{provider}|{operation}|{canonical}"
        tokens = token_set(query)
        with self._lock:
            stored = self._keys.get(payload_key)
            if stored is not None:
                return self._store.get(stored)
            if len(tokens) < self.min_tokens:
                return None
            best_match: Optional[str] = None
            best_similarity = 0.0
            for stored_key in self._keys:
                parts = stored_key.split("|", 2)
                if len(parts) != 3:
                    continue
                stored_provider_op = f"{parts[0]}|{parts[1]}"
                stored_canonical = parts[2]
                if stored_provider_op != f"{provider}|{operation}":
                    continue
                if not stored_canonical:
                    continue
                similarity = jaccard_similarity(canonical, stored_canonical)
                if similarity >= self.threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = stored_key
            if best_match is not None:
                return self._store.get(best_match)
        return None

    def store(self, provider: str, operation: str, query: str, payload: Dict) -> None:
        canonical = normalize_query(query)
        if not canonical:
            return
        payload_key = f"{provider}|{operation}|{canonical}"
        with self._lock:
            self._store[payload_key] = payload
            self._keys[payload_key] = payload_key

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._keys.clear()