import shutil

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.schemas import DeleteResponse, DocumentInfo, DocumentListResponse
from app.services.vectorstore import delete_by_doc_id, list_documents

router = APIRouter(prefix="/api", tags=["documents"])


@router.get("/documents", response_model=DocumentListResponse)
async def get_documents():
    docs = list_documents()
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs]
    )


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    deleted_count = delete_by_doc_id(doc_id)
    upload_dir = settings.uploads_path / doc_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir)

    if deleted_count == 0 and not upload_dir.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    return DeleteResponse(
        doc_id=doc_id,
        deleted=True,
        message=f"Removed {deleted_count} chunks and upload files.",
    )
