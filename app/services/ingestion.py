import hashlib
import re
import uuid
from pathlib import Path

from fastapi import UploadFile
from langchain_core.documents import Document

from app.config import settings
from app.logging_config import get_logger
from app.services.chunking import CHUNK_OVERLAP, CHUNK_SIZE, chunk_documents
from app.services.image_processor import process_image
from app.services.loaders import load_document
from app.services.vectorstore import add_documents, delete_by_doc_id, find_by_hash

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
logger = get_logger("ingestion")


def _describe_file_type(extension: str) -> str:
    if is_image(extension):
        return "Image"
    return {
        ".pdf": "PDF",
        ".docx": "DOCX",
        ".txt": "Text",
        ".md": "Markdown",
        ".markdown": "Markdown",
    }.get(extension.lower(), "Unknown")


def _format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def sanitize_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^\w.\-]", "_", name)
    return name or "upload"


def is_image(ext: str) -> bool:
    return ext.lower() in IMAGE_EXTENSIONS


async def ingest_file(file: UploadFile) -> dict:
    if not file.filename:
        raise ValueError("Filename is required")

    filename = sanitize_filename(file.filename)
    ext = Path(filename).suffix.lower()

    if ext not in settings.allowed_extensions:
        raise ValueError(f"File type not allowed: {ext}")

    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise ValueError(f"File exceeds {settings.max_upload_mb} MB limit")

    logger.info(
        "=== [DOCUMENT LOADING] filename=%s | size=%s | type=%s ===",
        filename,
        _format_file_size(len(content)),
        _describe_file_type(ext),
    )

    file_hash = hashlib.sha256(content).hexdigest()
    existing_doc_id = find_by_hash(file_hash)
    if existing_doc_id:
        logger.info(
            "[DOCUMENT LOADING] Duplicate detected; already indexed as document_id=%s",
            existing_doc_id,
        )
        return {
            "doc_id": existing_doc_id,
            "filename": filename,
            "chunks_added": 0,
            "status": "already indexed",
        }

    doc_id = str(uuid.uuid4())
    doc_dir = settings.uploads_path / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    file_path = doc_dir / filename
    file_path.write_bytes(content)
    logger.info("[DOCUMENT LOADING] Saved upload to %s", file_path)

    if is_image(ext):
        raw_docs = process_image(file_path, doc_id, filename)
    else:
        raw_docs = load_document(file_path, doc_id, filename)

    if not raw_docs:
        raise ValueError("No text could be extracted from the file")

    character_count = sum(len(document.page_content) for document in raw_docs)
    logger.info(
        "=== [CHUNKING] source_characters=%s | chunk_size=%s | overlap=%s ===",
        character_count,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
    chunks = chunk_documents(raw_docs)
    if not chunks:
        raise ValueError("No chunks produced from the file")
    logger.info(
        "[CHUNKING] Split %s into %s chunks with chunk_size=%s, overlap=%s",
        filename,
        len(chunks),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
        
    for chunk in chunks:
        if chunk.metadata is None:
            chunk.metadata = {}
        chunk.metadata["file_hash"] = file_hash

    delete_by_doc_id(doc_id)

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    logger.info(
        "=== [EMBEDDING & VECTOR DB] Generating embeddings for %s chunks ===",
        len(chunks),
    )
    add_documents(chunks, ids)
    logger.info(
        "[EMBEDDING & VECTOR DB] Successfully persisted %s vectors to ChromaDB collection '%s'",
        len(chunks),
        settings.collection_name,
    )

    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks_added": len(chunks),
        "status": "indexed",
    }
