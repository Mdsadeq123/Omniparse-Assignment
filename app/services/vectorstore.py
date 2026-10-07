from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.logging_config import get_logger
from app.services.embeddings import get_embeddings

_vectorstore: Chroma | None = None
logger = get_logger("vectorstore")


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[EMBEDDING & VECTOR DB] Opening ChromaDB collection '%s' at %s",
            settings.collection_name,
            settings.chroma_path,
        )
        _vectorstore = Chroma(
            collection_name=settings.collection_name,
            embedding_function=get_embeddings(),
            persist_directory=str(settings.chroma_path),
        )
    return _vectorstore


def add_documents(documents: list[Document], ids: list[str]) -> None:
    store = get_vectorstore()
    store.add_documents(documents=documents, ids=ids)
    logger.info(
        "[EMBEDDING & VECTOR DB] ChromaDB write complete; persistent store=%s",
        settings.chroma_path,
    )
    # Avoid serving results calculated before the new chunks were indexed.
    from app.services.rag import clear_retrieval_cache
    clear_retrieval_cache()


def find_by_hash(file_hash: str) -> str | None:
    store = get_vectorstore()
    collection = store._collection
    results = collection.get(where={"file_hash": file_hash})
    if results and results.get("metadatas") and len(results["metadatas"]) > 0:
        return results["metadatas"][0].get("doc_id")
    return None


def delete_by_doc_id(doc_id: str) -> int:
    store = get_vectorstore()
    collection = store._collection
    results = collection.get(where={"doc_id": doc_id})
    if not results or not results.get("ids"):
        return 0
    ids = results["ids"]
    collection.delete(ids=ids)
    from app.services.rag import clear_retrieval_cache
    clear_retrieval_cache()
    return len(ids)


def list_documents() -> list[dict]:
    store = get_vectorstore()
    collection = store._collection
    results = collection.get(include=["metadatas"])
    if not results or not results.get("metadatas"):
        return []

    by_doc: dict[str, dict] = {}
    for meta in results["metadatas"]:
        if not meta:
            continue
        doc_id = meta.get("doc_id", "")
        if not doc_id:
            continue
        if doc_id not in by_doc:
            by_doc[doc_id] = {
                "doc_id": doc_id,
                "filename": meta.get("source", "unknown"),
                "doc_type": meta.get("type", "document"),
                "chunk_count": 0,
            }
        by_doc[doc_id]["chunk_count"] += 1

    return list(by_doc.values())
