"""OpenRouter embeddings with Nemotron VL multimodal input format support."""

from __future__ import annotations

import httpx
from langchain_core.embeddings import Embeddings

from app.config import settings
from app.logging_config import get_logger
from app.services.openrouter import get_openrouter_headers

MAX_EMBED_CHARS = 8000
DEFAULT_BATCH_SIZE = 4
logger = get_logger("embeddings")


def _uses_multimodal_embed_format(model: str) -> bool:
    m = model.lower()
    return "embed-vl" in m or "nemotron-embed" in m


class OpenRouterEmbeddings(Embeddings):
    """Calls OpenRouter /embeddings with correct format for Nemotron VL models."""

    def __init__(
        self,
        model: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        if not settings.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self.model = model or settings.embedding_model
        self.batch_size = batch_size
        self.base_url = settings.openrouter_base_url.rstrip("/")
        self.multimodal = _uses_multimodal_embed_format(self.model)
        self._headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            **get_openrouter_headers(),
        }
        # Creating a client per query forces a TCP/TLS connection setup on every
        # retrieval. A single client reuses keep-alive connections instead.
        self._client = httpx.Client(timeout=180.0)

    def _truncate(self, text: str) -> str:
        text = text.strip()
        if len(text) > MAX_EMBED_CHARS:
            return text[:MAX_EMBED_CHARS]
        return text

    def _build_input_payload(self, texts: list[str], multimodal: bool) -> list | str:
        cleaned = [self._truncate(t) for t in texts]
        if multimodal:
            return [
                {"content": [{"type": "text", "text": t or " "}]}
                for t in cleaned
            ]
        return cleaned if len(cleaned) > 1 else (cleaned[0] if cleaned else " ")

    def _parse_response(self, data: dict, expected: int) -> list[list[float]]:
        if data.get("error"):
            err = data["error"]
            msg = err.get("message", err) if isinstance(err, dict) else str(err)
            raise ValueError(f"OpenRouter embedding error: {msg}")

        items = data.get("data") or []
        if not items:
            raise ValueError("No embedding data received from OpenRouter")

        items.sort(key=lambda x: x.get("index", 0))
        embeddings: list[list[float]] = []
        for item in items:
            emb = item.get("embedding")
            if not emb:
                raise ValueError("OpenRouter returned an empty embedding vector")
            embeddings.append(emb)

        if len(embeddings) != expected:
            raise ValueError(
                f"Expected {expected} embeddings, got {len(embeddings)}"
            )
        return embeddings

    def _call_api(self, texts: list[str], multimodal: bool) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": self._build_input_payload(texts, multimodal),
            "encoding_format": "float",
        }

        response = self._client.post(
            f"{self.base_url}/embeddings",
            headers=self._headers,
            json=payload,
        )

        if response.status_code != 200:
            detail = response.text[:800]
            raise ValueError(
                f"Embedding API HTTP {response.status_code}: {detail}"
            )

        return self._parse_response(response.json(), len(texts))

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        non_empty = [t if t.strip() else " " for t in texts]

        try:
            return self._call_api(non_empty, multimodal=self.multimodal)
        except ValueError as first_error:
            if not self.multimodal:
                raise first_error
            try:
                return self._call_api(non_empty, multimodal=False)
            except ValueError:
                raise first_error from None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        batch_count = (len(texts) + self.batch_size - 1) // self.batch_size
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_number = i // self.batch_size + 1
            logger.info(
                "[EMBEDDING & VECTOR DB] Embedding batch %s/%s (%s chunks)",
                batch_number,
                batch_count,
                len(batch),
            )
            all_embeddings.extend(self._embed_batch(batch))
        logger.info(
            "[EMBEDDING & VECTOR DB] Embedding generation complete: %s vectors",
            len(all_embeddings),
        )
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]


def get_embeddings() -> OpenRouterEmbeddings:
    return OpenRouterEmbeddings()
