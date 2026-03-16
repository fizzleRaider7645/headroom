from __future__ import annotations

import re
from typing import TYPE_CHECKING

from headroom.strategies.base import BaseStrategy, SessionContext

if TYPE_CHECKING:
    from headroom.core.budget import TokenBudget
    from headroom.core.message import TrackedMessage

# Common English stopwords to exclude from keyword matching
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can i you he she it we they "
    "this that these those and or but not of in on at to for with by from "
    "as if so then than when where who what how".split()
)


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _message_text(msg: "TrackedMessage") -> str:
    if isinstance(msg.content, str):
        return msg.content
    parts = []
    for block in msg.content:
        if isinstance(block, dict):
            parts.append(block.get("text", ""))
    return " ".join(parts)


class RelevanceFilterStrategy(BaseStrategy):
    """
    Drops messages whose content is not relevant to the current conversation topic.

    By default uses keyword overlap scoring (no extra dependencies).
    Set use_embeddings=True to use cosine similarity (requires sentence-transformers).
    """

    priority = 20

    def __init__(
        self,
        query_window: int = 3,
        similarity_threshold: float = 0.15,
        min_keep: int = 4,
        use_embeddings: bool = False,
    ):
        self.query_window = query_window
        self.similarity_threshold = similarity_threshold
        self.min_keep = min_keep
        self.use_embeddings = use_embeddings
        self._embedder = None  # lazy-loaded if use_embeddings=True

    @property
    def params(self) -> list[dict]:
        return [
            {
                "name": "query_window",
                "label": "Query window",
                "type": "int",
                "value": self.query_window,
                "min": 1,
                "max": 10,
            },
            {
                "name": "similarity_threshold",
                "label": "Similarity threshold",
                "type": "float",
                "value": self.similarity_threshold,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
            },
            {
                "name": "min_keep",
                "label": "Min keep",
                "type": "int",
                "value": self.min_keep,
                "min": 1,
                "max": 20,
            },
        ]

    def apply(
        self,
        messages: list["TrackedMessage"],
        budget: "TokenBudget",
        used_tokens: int,
        ctx: SessionContext,
    ) -> list["TrackedMessage"]:
        if len(messages) <= self.min_keep:
            return messages

        if self.use_embeddings:
            return self._apply_embeddings(messages)
        return self._apply_keywords(messages)

    def _apply_keywords(
        self, messages: list["TrackedMessage"]
    ) -> list["TrackedMessage"]:
        # Build query from last N user messages
        user_messages = [m for m in messages if m.role == "user"]
        query_msgs = user_messages[-self.query_window :]
        query_text = " ".join(_message_text(m) for m in query_msgs)
        query_kw = _keywords(query_text)

        if not query_kw:
            return messages  # can't score, keep all

        # Score each message; pinned messages always score 1.0
        scored: list[tuple[float, "TrackedMessage"]] = []
        for msg in messages:
            if msg.pinned:
                scored.append((1.0, msg))
                continue
            msg_kw = _keywords(_message_text(msg))
            if not msg_kw:
                scored.append((0.0, msg))
                continue
            overlap = len(query_kw & msg_kw)
            score = overlap / len(query_kw | msg_kw)  # Jaccard similarity
            scored.append((score, msg))

        # Always keep recent min_keep messages
        always_keep = set(id(m) for m in messages[-self.min_keep :])

        result = []
        for score, msg in scored:
            if (
                id(msg) in always_keep
                or score >= self.similarity_threshold
                or msg.pinned
            ):
                result.append(msg)

        # Guarantee at least min_keep messages
        if len(result) < self.min_keep:
            result = messages[-self.min_keep :]

        return result

    def _apply_embeddings(
        self, messages: list["TrackedMessage"]
    ) -> list["TrackedMessage"]:
        try:
            from sentence_transformers import SentenceTransformer, util  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Install sentence-transformers to use embedding-based relevance: "
                "pip install headroom[embeddings]"
            ) from e

        if self._embedder is None:
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

        user_messages = [m for m in messages if m.role == "user"]
        query_msgs = user_messages[-self.query_window :]
        query_text = " ".join(_message_text(m) for m in query_msgs)

        query_emb = self._embedder.encode(query_text, convert_to_tensor=True)
        texts = [_message_text(m) for m in messages]
        msg_embs = self._embedder.encode(texts, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, msg_embs)[0].tolist()

        always_keep = set(id(m) for m in messages[-self.min_keep :])
        result = []
        for score, msg in zip(scores, messages):
            if (
                id(msg) in always_keep
                or score >= self.similarity_threshold
                or msg.pinned
            ):
                result.append(msg)

        if len(result) < self.min_keep:
            result = messages[-self.min_keep :]

        return result
