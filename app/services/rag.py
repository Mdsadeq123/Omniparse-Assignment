from collections import OrderedDict
from threading import Lock
from time import monotonic

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.config import settings
from app.models.schemas import ChatResponse, SourceCitation
from app.services.openrouter import get_openrouter_client_kwargs
from app.services.vectorstore import get_vectorstore

NO_RELEVANT_ANSWER = "No relevant information found in your documents."

SYSTEM_PROMPT = """You answer using the context below.
If the context discusses the topic, answer from it even when the wording differs from the question.
Only if the question is clearly unrelated to the context (for example general trivia, small talk, or a different subject), reply with exactly this sentence and nothing else:
No relevant information found in your documents.
Do not guess from world knowledge. Do NOT use numerical citations like [1] or [2]. When you answer from the context, cite sources using [filename, page/section]. Be concise and accurate."""


# Results are immutable LangChain Documents, so they can be safely reused between
# requests. The cache is bounded and expires quickly so new uploads become visible.
_retrieval_cache: OrderedDict[tuple[str, int], tuple[float, list[tuple[Document, float]]]] = OrderedDict()
_retrieval_cache_lock = Lock()


def _cache_key(message: str, k: int) -> tuple[str, int]:
    """Normalize inconsequential whitespace without altering the user's question."""
    return (" ".join(message.split()).casefold(), k)


def retrieve_documents(message: str, k: int) -> list[tuple[Document, float]]:
    """Retrieve top-k documents with a small LRU/TTL cache for repeat queries."""
    key = _cache_key(message, k)
    now = monotonic()
    ttl = settings.retrieval_cache_ttl_seconds

    with _retrieval_cache_lock:
        cached = _retrieval_cache.get(key)
        if cached and now - cached[0] < ttl:
            _retrieval_cache.move_to_end(key)
            return cached[1]
        if cached:
            del _retrieval_cache[key]

    # Chroma uses its ANN index here rather than scanning every stored chunk.
    results = get_vectorstore().similarity_search_with_score(message, k=k)

    if settings.retrieval_cache_size > 0 and ttl > 0:
        with _retrieval_cache_lock:
            _retrieval_cache[key] = (now, results)
            _retrieval_cache.move_to_end(key)
            while len(_retrieval_cache) > settings.retrieval_cache_size:
                _retrieval_cache.popitem(last=False)
    return results


def clear_retrieval_cache() -> None:
    """Invalidate cached search results after the indexed corpus changes."""
    with _retrieval_cache_lock:
        _retrieval_cache.clear()


def _relevant_results(
    results: list[tuple[Document, float]],
) -> list[tuple[Document, float]]:
    """Optionally drop far chunks. Disabled when max_retrieval_distance <= 0."""
    max_distance = settings.max_retrieval_distance
    if max_distance <= 0:
        return results
    kept: list[tuple[Document, float]] = []
    for doc, score in results:
        if score is None or float(score) <= max_distance:
            kept.append((doc, score))
    return kept


def _normalize_answer(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "".join(parts).strip()
    return str(content).strip()


def _is_no_relevant_answer(answer: str) -> bool:
    text = " ".join(answer.lower().split()).strip(" .!'\"")
    if not text:
        return True
    if text == "no relevant information found in your documents":
        return True
    if text.startswith("no relevant information found"):
        return True
    short_abstentions = {
        "i don't know",
        "i do not know",
        "don't know",
        "do not know",
    }
    return text in short_abstentions


def _empty_relevant_response() -> ChatResponse:
    return ChatResponse(answer=NO_RELEVANT_ANSWER, sources=[])


def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        source = meta.get("source", "unknown")
        page = meta.get("page", "")
        doc_type = meta.get("type", "document")
        header = f"[{i}] {source}"
        if page:
            header += f" (page {page})"
        header += f" [{doc_type}]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def chat(message: str, top_k: int | None = None) -> ChatResponse:
    k = top_k or settings.top_k
    results = retrieve_documents(message, k)

    if not results:
        return ChatResponse(
            answer="No indexed documents found. Upload files first.",
            sources=[],
        )

    results = _relevant_results(results)
    if not results:
        return _empty_relevant_response()

    docs = [doc for doc, _ in results]
    context = _format_context(docs)

    llm = ChatOpenAI(
        model=settings.chat_model,
        temperature=0.2,
        **get_openrouter_client_kwargs(),
    )

    try:
        response = llm.invoke(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    f"Context:\n{context}\n\nQuestion:\n{message}",
                ),
            ]
        )
    except Exception as e:
        error_str = str(e)
        if "502" in error_str or "429" in error_str:
            raise ValueError("Service is temporarily busy. Please try again in a few seconds.") from e
        raise e

    answer = _normalize_answer(
        response.content if hasattr(response, "content") else response
    )
    if _is_no_relevant_answer(answer):
        return _empty_relevant_response()

    sources: list[SourceCitation] = []
    for doc, score in results:
        meta = doc.metadata or {}
        snippet = doc.page_content[:300]
        if len(doc.page_content) > 300:
            snippet += "..."
        sources.append(
            SourceCitation(
                file=meta.get("source", "unknown"),
                snippet=snippet,
                score=round(float(score), 4) if score is not None else None,
                page=meta.get("page"),
                doc_type=meta.get("type"),
            )
        )

    return ChatResponse(answer=answer, sources=sources)
